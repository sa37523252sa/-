const pool = require("../config/db");
const cloudinary = require("cloudinary").v2;
const fs = require("fs");

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET
});

exports.addProduct = async (req, res) => {
  try {
    const { title, options } = req.body;
    let image_url = "";

    if (req.file) {
      const result = await cloudinary.uploader.upload(req.file.path);
      image_url = result.secure_url;
      fs.unlinkSync(req.file.path);
    }

    const product = await pool.query(
      "INSERT INTO products(title, image_url) VALUES ($1,$2) RETURNING id",
      [title, image_url]
    );
    const productId = product.rows[0].id;

    if (options && Array.isArray(options)) {
      for (const op of options) {
        await pool.query(
          "INSERT INTO product_options(product_id, option_name) VALUES ($1,$2)",
          [productId, op]
        );
      }
    }

    res.json({ productId, message: "新增成功" });
  } catch (err) {
    console.log(err);
    res.status(500).json({ message: "新增商品失敗" });
  }
};

exports.listProducts = async (req, res) => {
  const products = await pool.query("SELECT * FROM products ORDER BY id DESC");
  res.json(products.rows);
};

exports.searchProducts = async (req, res) => {
  const { keyword } = req.query;
  const products = await pool.query(
    "SELECT * FROM products WHERE title ILIKE $1",
    [`%${keyword}%`]
  );
  res.json(products.rows);
};