use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

static GLOBAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

impl MetalContext {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let command_queue = device.newCommandQueue().expect("Failed to create command queue");
        Self {
            device,
            command_queue,
        }
    }

    pub fn global() -> &'static MetalContext {
        GLOBAL_CONTEXT.get_or_init(MetalContext::new)
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }
}
