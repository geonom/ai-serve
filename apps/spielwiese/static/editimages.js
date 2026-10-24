// The Bild tab's Edit mode takes more than the image to edit (2026-10-24, owner: "a background and up to four more
// pictures, e.g. a group photo of four fictional characters"): picture 1 = the image to edit or the scene, then up to
// max_images - 1 more (the model's catalog row, GET /v1/images/models?task=edit). Qwen-Image-Edit 2511 sees pictures
// 2 and 3 on their own; three or four more are put side by side into one picture 2 by image-serve (left to right).
// The output shape: like picture 1, or one of the model's aspect ratios. Pure (tests: tests/editimages.test.mjs).
const EditImages = (() => {
    const NATIVE = 2;   // extra pictures the model sees one by one; more go side by side into picture 2
    const ORDINAL = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'];

    function extraLimit(maxImages) {
        return Math.max(0, (Number.isInteger(maxImages) ? maxImages : 1) - 1);
    }
    // the list after adding `incoming` (in order) to `list`, at most `limit`; dropped = how many didn't fit
    function add(list, incoming, limit) {
        const room = Math.max(0, limit - list.length);
        return { list: [...list, ...incoming.slice(0, room)], dropped: Math.max(0, incoming.length - room) };
    }
    function labels(count) {
        if (count <= NATIVE) return Array.from({ length: count }, (_, i) => tx('Picture {n}', { n: i + 2 }));
        return Array.from({ length: count }, (_, i) => tx('Picture 2 ({nth} from left)', { nth: tx(ORDINAL[i]) }));
    }
    function hint(count) {
        if (count === 0) return '';
        if (count === 1) return tx('In the prompt: picture 1 is the image above, picture 2 the extra one – e.g. "the creature from picture 2 in the scene of picture 1".');
        if (count <= NATIVE) return tx('In the prompt: picture 1 is the image above, then picture 2 and picture 3 – e.g. "the two figures from picture 2 and picture 3 in the scene of picture 1".');
        return tx('These pictures are put side by side into one picture 2 (left to right) – e.g. "a group photo of the {n} characters from picture 2 in the scene of picture 1", and describe each one briefly.', { n: count });
    }
    function request(extra, aspect) {
        const out = {};
        if (extra.length) out.extra_images_base64 = [...extra];
        if (aspect) out.aspect_ratio = aspect;
        return out;
    }
    // the Output shape choices: '' = like picture 1, then the model's aspect ratios
    function aspects(modelAspects) {
        return ['', ...(Array.isArray(modelAspects) ? modelAspects : [])];
    }
    return { extraLimit, add, labels, hint, request, aspects };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = EditImages;
