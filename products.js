const express = require("express");
const router = express.Router();

// 修正路徑：middleware 不在資料夾，是同一層
const { auth, admin } = require("./authMiddleware");

// 修正路徑：productController 也在同層
const { addProduct, listProducts, searchProducts } = require("./productController");

const multer = require("multer");
const upload = multer({ dest: "uploads/" });

// 新增商品（需要登入 + 管理員）
router.post("/add", auth, admin, upload.single("image"), addProduct);

// 列出商品
router.get("/list", listProducts);

// 搜尋商品
router.get("/search", searchProducts);

module.exports = router;
