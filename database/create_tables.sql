CREATE TABLE IF NOT EXISTS regions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS publics (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    region_id INT REFERENCES regions(id) ON DELETE SET NULL
);

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    post_id INT NOT NULL,
    public_id INT REFERENCES publics(id) ON DELETE CASCADE,
    post_text TEXT,
    post_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttype VARCHAR(50) DEFAULT NULL,
    views INT DEFAULT 0,
    likes INT DEFAULT 0,
    comments INT DEFAULT 0,
    reposts INT DEFAULT 0,
    UNIQUE (post_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    post_id INT REFERENCES posts(post_id) ON DELETE CASCADE,
    embedding vector(1000)
);

CREATE TABLE IF NOT EXISTS last_upds (
    id SERIAL PRIMARY KEY,
    public_id INT,
    update_date VARCHAR(12)
);

CREATE TABLE IF NOT EXISTS model_output (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    related_posts INT[],
    update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    post_count INT DEFAULT 0,
    avg_embedding vector(1000)
);
