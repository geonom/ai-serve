// Microphone frames for "Speech > Talk" (talk.js): hands each block of samples to the page.
class TalkCapture extends AudioWorkletProcessor {
    process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch && ch.length) this.port.postMessage(ch.slice(0));
        return true;
    }
}
registerProcessor('talk-capture', TalkCapture);
