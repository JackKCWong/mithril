use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use tract_onnx::{
    prelude::{tract_itertools::Itertools, *},
    tract_core::anyhow::Ok,
};

fn main() -> Result<(), TractError> {
    let model_path = Path::join(&PathBuf::from_str("./model")?, "silero_vad.onnx");

    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_runnable()?;

    let mut reader = hound::WavReader::open("en_example.wav")?;
    let spec = reader.spec();
    println!("{}, {}, {}", spec.bits_per_sample, spec.channels, spec.sample_rate);

    let samples = reader.samples::<i32>().map(|x| x.unwrap() as f32).collect_vec();
    let s: Tensor = tract_ndarray::Array2::from_shape_vec((1, samples.len()), samples)?.into();
    let h: Tensor = tract_ndarray::Array3::<f32>::zeros((2, 1, 64)).into();
    let c: Tensor = tract_ndarray::Array3::<f32>::zeros((2, 1, 64)).into();
    let sr: Tensor = tract_ndarray::Array1::<i64>::from_shape_vec(1, vec![16000])?.into();

    let _output = model.run(tvec!(s.into(), sr.into(), h.into(), c.into()))?;

    Ok(())
}
