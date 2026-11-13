use gramgrid_proto::mesh::MeshEnvelope;

fn model_wire() -> Vec<u8> {
    // Independent Python struct + zlib fixture, not the production encoder.
    let hex = "474730320200050009000000000000000000000000000000000001000000000000000000600000000100000002000000b2aac5cf0300000007000000000000007b2270726f6d7074223a2266697874757265222c226d61785f746f6b656e73223a31362c226d6f64656c5f7265766973696f6e223a2231313131313131313131313131313131313131313131313131313131313131313131313131313131227d";
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn a_whole_model_query_has_an_explicit_opcode_and_opaque_payload() {
    let wire = model_wire();
    let decoded = MeshEnvelope::decode(&wire).expect("GG02 cannot decode the explicit whole-model query opcode");
    assert_eq!(decoded.op as u16, 5);
    assert_eq!(decoded.total_steps, 1);
    assert_eq!(decoded.encode().unwrap(), wire);
}

#[test]
fn a_whole_model_query_cannot_pretend_to_be_multiple_layer_stages() {
    let mut decoded = MeshEnvelope::decode(&model_wire()).unwrap();
    decoded.total_steps = 2;
    assert!(decoded.encode().is_err(), "full model query falsely advertised a layer chain");
}
