use eyre::{bail, Result};
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

use crate::{session, vad_result::VadResult};

#[derive(Debug)]
pub struct Vad {
    session: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
}

impl Vad {
    pub fn new<P: AsRef<Path>>(model_path: P, sample_rate: usize) -> Result<Self> {
        if ![8000_usize, 16000].contains(&sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let session = session::create_session(model_path)?;
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));
        let sample_rate_tensor = Array1::from_vec(vec![sample_rate as i64]);

        Ok(Self {
            session,
            h_tensor,
            c_tensor,
            sample_rate_tensor,
        })
    }

    pub fn compute(&mut self, samples: &[f32]) -> Result<VadResult> {
        let samples_tensor = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;

        // Convert ndarray to Vec and create Value
        let input_shape: Vec<i64> = samples_tensor.shape().iter().map(|&x| x as i64).collect();
        let input_data: Vec<f32> = samples_tensor.iter().cloned().collect();
        let input_value = Value::from_array((input_shape, input_data))?;

        let sr_shape: Vec<i64> = self
            .sample_rate_tensor
            .shape()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let sr_data: Vec<i64> = self.sample_rate_tensor.iter().cloned().collect();
        let sr_value = Value::from_array((sr_shape, sr_data))?;

        let h_shape: Vec<i64> = self.h_tensor.shape().iter().map(|&x| x as i64).collect();
        let h_data: Vec<f32> = self.h_tensor.iter().cloned().collect();
        let h_value = Value::from_array((h_shape, h_data))?;

        let c_shape: Vec<i64> = self.c_tensor.shape().iter().map(|&x| x as i64).collect();
        let c_data: Vec<f32> = self.c_tensor.iter().cloned().collect();
        let c_value = Value::from_array((c_shape, c_data))?;

        let result = self.session.run(ort::inputs![
            "input" => input_value,
            "sr" => sr_value,
            "h" => h_value,
            "c" => c_value
        ])?;

        // Update internal state tensors.
        let (_, h_data) = result
            .get("hn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        self.h_tensor = Array3::from_shape_vec((2, 1, 64), h_data.to_vec())
            .expect("Shape mismatch for h_tensor");

        let (_, c_data) = result
            .get("cn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        self.c_tensor = Array3::from_shape_vec((2, 1, 64), c_data.to_vec())
            .expect("Shape mismatch for c_tensor");

        let (_, prob_data) = result
            .get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        let prob = prob_data[0];
        Ok(VadResult { prob })
    }

    pub fn reset(&mut self) {
        self.h_tensor.fill(0.0);
        self.c_tensor.fill(0.0);
    }
}
