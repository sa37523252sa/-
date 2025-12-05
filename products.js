const express = require("express");
const router = express.Router();
const { auth, admin } = require("../middleware/authMiddleware");
const { addProduct, listProducts, searchProducts } = require("../controllers/productController");
const multer = require("multer");
const upload = multer({ dest: "uploads/" });

router.post("/add", auth, admin, upload.single("image"), addProduct);
router.get("/list", listProducts);
router.get("/search", searchProducts);

module.exports = router;