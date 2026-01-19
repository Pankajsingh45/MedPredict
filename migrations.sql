CREATE DATABASE medpredict;
USE medpredict;

CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(80) UNIQUE NOT NULL,
  email VARCHAR(120) UNIQUE NOT NULL,
  password_hash VARCHAR(256) NOT NULL,
  is_admin BOOLEAN DEFAULT FALSE
);

CREATE TABLE predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  input_features TEXT NOT NULL,
  predicted_label VARCHAR(80) NOT NULL,
  probabilities TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
