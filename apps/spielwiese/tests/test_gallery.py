"""Gallery rules: every person's images live in their own folder, ids never become paths, only real
image bytes and known metadata are stored, nothing of one person is reachable by another."""
import base64
import os
import tempfile
import unittest

from spielwiese.gallery import Gallery, GalleryError, decode_image, parse_meta, person_dir

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\0" * 32


class TestRules(unittest.TestCase):
    def test_person_dir_is_a_hash_of_the_account(self):
        a, b = person_dir("/data", "alice42"), person_dir("/data", "../../etc")
        self.assertTrue(a.startswith("/data/users/") and len(os.path.basename(a)) == 32)
        self.assertTrue(b.startswith("/data/users/") and "/etc" not in b[len("/data/users/"):])
        self.assertNotEqual(a, person_dir("/data", "someone-else"))

    def test_decode_image_accepts_png_jpeg_webp_only(self):
        self.assertEqual(decode_image(base64.b64encode(PNG).decode())[1], "png")
        self.assertEqual(decode_image("data:image/png;base64," + base64.b64encode(PNG).decode())[1], "png")
        self.assertEqual(decode_image(base64.b64encode(JPG).decode())[1], "jpg")
        for bad in ("", "not base64!!", base64.b64encode(b"<svg onload=alert(1)>").decode(),
                    base64.b64encode(b"GIF89a....").decode(), 42, None):
            with self.assertRaises(GalleryError):
                decode_image(bad)

    def test_decode_image_size_cap(self):
        with self.assertRaises(GalleryError):
            decode_image(base64.b64encode(PNG + b"\0" * (26 * 1024 * 1024)).decode())

    def test_meta_known_fields_only(self):
        m = parse_meta({"prompt": " a fox ", "model": "z-image-turbo", "label": "Z-Image Turbo", "steps": 8,
                        "seed": 7, "aspect_ratio": "16:9", "seconds": 9.5, "source": "bild"})
        self.assertEqual(m["prompt"], "a fox")
        self.assertEqual(parse_meta({}), {"source": "bild"})
        # labels are display text (shown via textContent): brackets and commas are fine, control characters not
        self.assertEqual(parse_meta({"label": "Qwen-Image 2.1 (offiziell, int8)"})["label"], "Qwen-Image 2.1 (offiziell, int8)")
        for bad_label in ("a\nb", "x" * 121, "", 5):
            with self.assertRaises(GalleryError, msg=bad_label):
                parse_meta({"label": bad_label})
        for bad in ({"evil": 1}, {"prompt": "x" * 4001}, {"steps": 0}, {"steps": "8"}, {"seed": -1},
                    {"seed": True}, {"aspect_ratio": "2:1"}, {"model": "../x"}, {"source": "admin"},
                    {"seconds": -1}, {"comparison": "../../x"}):
            with self.assertRaises(GalleryError, msg=bad):
                parse_meta(bad)


class TestGallery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.g = Gallery(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_list_get_delete(self):
        e1 = self.g.save("anna", PNG, {"prompt": "fox", "model": "iris-turbo"})
        e2 = self.g.save("anna", JPG, {"prompt": "owl"})
        self.assertEqual([e["id"] for e in self.g.list("anna")], [e2["id"], e1["id"]])   # newest first
        path, ctype = self.g.image("anna", e1["id"])
        self.assertEqual((open(path, "rb").read(), ctype), (PNG, "image/png"))
        self.assertTrue(self.g.delete("anna", e1["id"]))
        self.assertIsNone(self.g.image("anna", e1["id"]))
        self.assertEqual([e["id"] for e in self.g.list("anna")], [e2["id"]])

    def test_people_never_see_each_other(self):
        e = self.g.save("anna", PNG, {"prompt": "secret"})
        self.assertEqual(self.g.list("ben"), [])
        self.assertIsNone(self.g.image("ben", e["id"]))
        self.assertFalse(self.g.delete("ben", e["id"]))
        self.assertIsNotNone(self.g.image("anna", e["id"]))

    def test_ids_are_never_paths(self):
        self.g.save("anna", PNG, {})
        for bad in ("../x", "..", "/etc/passwd", "a" * 25 + "/..", "", "ABCDEF"):
            self.assertIsNone(self.g.image("anna", bad))
            self.assertFalse(self.g.delete("anna", bad))

    def test_list_paging_and_filter(self):
        ids = [self.g.save("anna", PNG, {"comparison": "c" * 25 if i % 2 else None} if i % 2 else {})["id"]
               for i in range(5)]
        page = self.g.list("anna", limit=2)
        self.assertEqual([e["id"] for e in page], ids[::-1][:2])
        self.assertEqual([e["id"] for e in self.g.list("anna", limit=10, before=page[-1]["id"])], ids[::-1][2:])
        self.assertEqual(len(self.g.list("anna", comparison="c" * 25)), 2)
