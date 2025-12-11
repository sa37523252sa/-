const pool = require("./db"); // 修正路徑
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

exports.register = async (req, res) => {
  const { email, name, phone, password } = req.body;
  try {
    const hash = await bcrypt.hash(password, 10);
    const result = await pool.query(
      "INSERT INTO users(email, name, phone, password) VALUES ($1,$2,$3,$4) RETURNING id",
      [email, name, phone, hash]
    );
    res.json({ userId: result.rows[0].id });
  } catch (err) {
    console.log(err);
    res.status(400).json({ message: "Email 已存在或錯誤" });
  }
};

exports.login = async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await pool.query("SELECT * FROM users WHERE email=$1", [email]);
    if (user.rowCount === 0)
      return res.status(400).json({ message: "帳號錯誤" });

    const ok = await bcrypt.compare(password, user.rows[0].password);
    if (!ok) return res.status(400).json({ message: "密碼錯誤" });

    const token = jwt.sign(
      {
        id: user.rows[0].id,
        email,
        is_admin: user.rows[0].is_admin,
      },
      process.env.JWT_SECRET
    );

    res.json({ token });
  } catch (err) {
    console.log(err);
    res.status(500).json({ message: "伺服器錯誤" });
  }
};
