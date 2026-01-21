use irontensor::{MetalContext, Precision, Tensor};
use objc2_metal::MTLDevice;

fn main() {
    // Initialize the Metal context
    let ctx = MetalContext::global();
    println!("Metal device: {}", ctx.device().name());

    // Create a tensor from f32 data
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2, 3];
    let tensor = Tensor::from_f32_slice(&data, &shape);

    println!(
        "Created tensor with shape {:?}, {} elements",
        tensor.shape(),
        tensor.numel()
    );

    // Read back the data (unified memory - no copy needed)
    let readback = tensor.as_f32_slice();
    println!("Data: {:?}", readback);

    // Verify
    assert_eq!(readback, &data[..]);
    println!("Data matches!");

    // Create a zeroed tensor
    let zeros = Tensor::zeros(&[4, 4], Precision::FP32);
    println!(
        "Created zeros tensor with shape {:?}, byte_size={}",
        zeros.shape(),
        zeros.byte_size()
    );

    // Verify zeros
    let zeros_data = zeros.as_f32_slice();
    assert!(zeros_data.iter().all(|&x| x == 0.0));
    println!("Zeros verified!");
}
