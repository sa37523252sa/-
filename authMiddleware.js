const jwt = require("jsonwebtoken");

exports.auth = (req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ message: "未登入" });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch {
    res.status(403).json({ message: "Token 錯誤" });
  }
};

exports.admin = (req, res, next) => {
  if (!req.user?.is_admin) {
    return res.status(403).json({ message: "需要管理員權限" });
  }
  next();
};