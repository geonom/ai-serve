// Which way a Create images "Generate" model goes (2026-11-27). Main-ComfyUI checkpoints (their catalog row names the
// checkpoint file as last alias) use /api/comfyui/generate with LoRA, resolution and negative prompt; everything else
// -- other plugins, and ComfyUI models without a checkpoint file such as MOSS-Image -- answers on
// /v1/images/generations by catalog id. Pure (tests: tests/imageroute.test.mjs).
const ImageRoute = (() => {
    function generateRoute(m) {
        const file = (m.aliases || []).slice(-1)[0];
        if (m.provider === 'comfyui' && file && /\.(gguf|safetensors|ckpt)$/i.test(file)) return { value: file, gateway: false };
        return { value: m.id, gateway: true };
    }
    return { generateRoute };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ImageRoute;
