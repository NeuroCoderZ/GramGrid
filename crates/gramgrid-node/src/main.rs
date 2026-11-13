mod resources;

use gramgrid_proto::{decode_add, decode_matmul, MatmulTask, ComputeResult, encode_result};
use serde_json::json;
use std::{ffi::{CStr,CString,c_void,c_char},time::Instant,path::PathBuf};
use tungstenite::{connect,Message};

unsafe extern "C" {
    fn gramgrid_create(image:*const c_char,instructions:*const c_char)->*mut c_void;
    fn gramgrid_create_matmul16(image:*const c_char,instructions:*const c_char)->*mut c_void;
    fn gramgrid_destroy(handle:*mut c_void);
    fn gramgrid_device_name(handle:*mut c_void)->*const c_char;
    fn gramgrid_last_error()->*const c_char;
    fn gramgrid_add_i32(handle:*mut c_void,a:*const i32,b:*const i32,out:*mut i32,count:u32)->i32;
    fn gramgrid_matmul16_i32(handle:*mut c_void,a:*const i32,b:*const i32,out:*mut i32)->i32;
}
#[derive(Clone,Copy,PartialEq)]
enum Mode { Add, Matmul16 }
struct Npu {handle:*mut c_void,mode:Mode}
impl Drop for Npu {fn drop(&mut self){unsafe{gramgrid_destroy(self.handle);}}}
fn native_error()->String {unsafe{CStr::from_ptr(gramgrid_last_error())}.to_string_lossy().into_owned()}
impl Npu {
    fn new(root:PathBuf,mode:Mode)->Result<Self,String> {
        let stem=if mode==Mode::Add {"add_i32"} else {"matmul16"};
        let image=CString::new(root.join(format!("native/xdna/build/{stem}.xclbin")).to_string_lossy().as_bytes()).unwrap();
        let inst=CString::new(root.join(format!("native/xdna/build/{stem}.insts.bin")).to_string_lossy().as_bytes()).unwrap();
        let handle=unsafe{if mode==Mode::Add {gramgrid_create(image.as_ptr(),inst.as_ptr())}
            else {gramgrid_create_matmul16(image.as_ptr(),inst.as_ptr())}};
        if handle.is_null(){return Err(native_error());}
        Ok(Self{handle,mode})
    }
    fn name(&self)->String {unsafe{CStr::from_ptr(gramgrid_device_name(self.handle))}.to_string_lossy().into_owned()}
    fn add(&self,a:&[i32],b:&[i32])->Result<Vec<i32>,String> {
        if self.mode!=Mode::Add||a.len()!=b.len()||a.is_empty()||a.len()>65536{return Err("неверные векторы".into());}
        let mut output=vec![0;a.len()];
        let status=unsafe{gramgrid_add_i32(self.handle,a.as_ptr(),b.as_ptr(),output.as_mut_ptr(),a.len() as u32)};
        if status!=0{return Err(native_error());}
        Ok(output)
    }
    fn matrix(&self,task:&MatmulTask)->Result<Vec<i32>,String> {
        if self.mode!=Mode::Matmul16||task.m>16||task.k>16||task.n>16{return Err("матрица превышает объявленный предел NPU 16".into());}
        let mut a=vec![0i32;256];let mut b=vec![0i32;256];let mut c=vec![0i32;256];
        for r in 0..task.m as usize {for k in 0..task.k as usize {a[r*16+k]=task.a[r*task.k as usize+k] as i32;}}
        for k in 0..task.k as usize {for n in 0..task.n as usize {b[k*16+n]=task.b[k*task.n as usize+n] as i32;}}
        let status=unsafe{gramgrid_matmul16_i32(self.handle,a.as_ptr(),b.as_ptr(),c.as_mut_ptr())};
        if status!=0{return Err(native_error());}
        let mut output=Vec::with_capacity(task.m as usize*task.n as usize);
        for r in 0..task.m as usize {output.extend_from_slice(&c[r*16..r*16+task.n as usize]);}
        Ok(output)
    }
    fn selftest(&self)->Result<(),String> {
        if self.mode==Mode::Add {
            let a=[i32::MAX,i32::MIN,7];let b=[1,-1,-9];let output=self.add(&a,&b)?;
            if !a.iter().zip(b).zip(output).all(|((a,b),c)|a.wrapping_add(b)==c){return Err("аппаратная самопроверка не прошла".into());}
        } else {
            let a=(0..256).map(|i|if i/16==i%16 {1i8}else{0}).collect();
            let b:Vec<i8>=(0..256).map(|i|(i%17) as i8-8).collect();
            let task=MatmulTask{id:1,attempt:0,m:16,k:16,n:16,a,b};
            if self.matrix(&task)?!=task.b.iter().map(|x|*x as i32).collect::<Vec<_>>(){return Err("аппаратная самопроверка матрицы не прошла".into());}
        }
        Ok(())
    }
}
enum SessionEnd { Retry, Fatal(String) }
fn run_session(npu:&Npu,address:&str,id:&str)->Result<SessionEnd,tungstenite::Error> {
    let limits=match resources::limits(){Ok(limits)=>limits,Err(e)=>return Ok(SessionEnd::Fatal(e))};
    let sample=resources::sample();
    if let Err(e)=gramgrid_proto::admission::check(true,sample,limits){return Ok(SessionEnd::Fatal(e.into()));}
    let (mut socket,_)=connect(address)?;
    let op=if npu.mode==Mode::Add {2}else{1};
    let mut device=json!({"name":npu.name(),"runtime":"XRT 2.25","software":false,"local_selftest":true,
        "kernel":if op==2{"ADD_I32"}else{"MATMUL16_I32"}});
    if op==1 {device["max_matmul_dims"]=json!([16,16,16]);}
    socket.send(Message::Text(json!({"type":"hello","id":id,"backend":"native-xdna","operators":[op],"device":device,"consent":true,
        "resources":resources::as_json(sample),"resource_limits":resources::limits_json(limits)}).to_string().into()))?;
    println!("GramGrid: {id} подключён, самопроверка настоящего NPU пройдена");
    loop {
        match socket.read()? {
            Message::Binary(bytes)=>{
                let started=Instant::now();
                if let Err(e)=gramgrid_proto::admission::check(true,resources::sample(),limits) {
                    socket.send(Message::Text(json!({"type":"failure","error":e,"category":"resource-limit"}).to_string().into()))?;
                    return Ok(SessionEnd::Fatal(e.into()));
                }
                let computed:Result<ComputeResult,String>=(||{
                    if npu.mode==Mode::Add {
                        let task=decode_add(&bytes).map_err(str::to_owned)?;
                        let values=npu.add(&task.a,&task.b)?;
                        Ok(ComputeResult{id:task.id,attempt:task.attempt,op:2,m:values.len() as u32,k:1,n:1,elapsed_ns:0,values})
                    } else {
                        let task=decode_matmul(&bytes).map_err(str::to_owned)?;
                        let values=npu.matrix(&task)?;
                        Ok(ComputeResult{id:task.id,attempt:task.attempt,op:1,m:task.m,k:task.k,n:task.n,elapsed_ns:0,values})
                    }
                })();
                let mut result=match computed {Ok(result)=>result,Err(error)=>{
                    socket.send(Message::Text(json!({"type":"failure","error":error}).to_string().into()))?;
                    return Ok(SessionEnd::Fatal(error));
                }};
                result.elapsed_ns=started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                let packet=match encode_result(&result){Ok(packet)=>packet,Err(error)=>return Ok(SessionEnd::Fatal(error.into()))};
                socket.send(Message::Binary(packet.into()))?;
            },
            Message::Close(_)=>return Ok(SessionEnd::Retry),
            Message::Ping(bytes)=>socket.send(Message::Pong(bytes))?,
            _=>{},
        }
    }
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize()?;
    let mode=match std::env::var("GRAMGRID_NODE_MODE").unwrap_or_else(|_|"add".into()).as_str(){
        "add"=>Mode::Add,"matmul16"=>Mode::Matmul16,_=>return Err("неизвестный GRAMGRID_NODE_MODE".into())};
    let limits=resources::limits().map_err(std::io::Error::other)?;
    let sample=resources::sample();
    if std::env::var("GRAMGRID_DUMP_RESOURCES").as_deref()==Ok("1") {
        println!("{}",json!({"resources":resources::as_json(sample),"limits":resources::limits_json(limits)}));return Ok(());
    }
    gramgrid_proto::admission::check(true,sample,limits).map_err(std::io::Error::other)?;
    let npu=Npu::new(root,mode).map_err(std::io::Error::other)?;npu.selftest().map_err(std::io::Error::other)?;
    let address=std::env::args().nth(1).unwrap_or_else(||"ws://127.0.0.1:8787/ws".into());
    let id=format!("xdna-{}",std::process::id());
    loop {
        match run_session(&npu,&address,&id) {
            Ok(SessionEnd::Fatal(reason))=>return Err(reason.into()),
            Ok(SessionEnd::Retry)=>eprintln!("Координатор закрыл соединение; восстанавливаю связь"),
            Err(error)=>eprintln!("Соединение потеряно: {error}; восстанавливаю связь"),
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
