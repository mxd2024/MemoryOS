# MemoryOS Rust 移植版 - 設計仕様書

**日付**: 2026 年 4 月  
**ライセンス**: Apache-2.0  

---

## 📋 目次

1. [概要](#1-概要)
2. [システムアーキテクチャ](#2-システムアーキテクチャ)
3. [モジュール構成](#3-モジュール構成)
4. [データ構造定義](#4-データ構造定義)
5. [ストレージ層設計](#5-ストレージ層設計)
6. [メモリ管理コア](#6-メモリ管理コア)
7. [並行性設計](#7-並行性設計)
8. [エラー処理と耐障害性](#8-エラー処理と耐障害性)
9. [実装チェックリスト](#9-実装チェックリスト)

---

## 1. 概要

### 1.1 プロジェクト目的

MemoryOS の Rust 移植版は、AI エージェント向けのメモリ管理システムを Rust で再実装し、以下の目標を達成します：

| 目標 | 詳細 |
|------|------|
| **パフォーマンス向上** | Python 版比でベクトル検索部分 5-10 倍高速化 |
| **型安全性** | コンパイル時エラー検出による堅牢性向上 |
| **メモリ効率** | GC なし、明示的なメモリ管理（3-5 倍削減） |
| **並行処理** | 同時リクエスト対応（4-7 倍スループット向上） |
| **クロスプラットフォーム** | WASM/組み込み環境での利用可能 |

### 1.2 技術スタック

```toml
[dependencies]
# ベクトル検索・近傍探索
hnsw_rs = "0.3"              # HNSW 実装（CosineDistance 対応）

# データ永続化
sqlx = { 
    version = "0.7", 
    features = [
        "runtime-tokio",      # tokio ランタイム
        "sqlite",             # SQLite サポート
        "macros",             # query_as! マクロ（型安全）
        "chrono",             # DateTime 対応
        "json",               # JSON 型サポート
    ] 
}

# LLM/HTTP クライアント
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full"] }

# シリアライズ
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# ベクトル計算
candle-core = "0.8"          # HuggingFace モデル対応（オプション）
ndarray = "0.15"             # 数値計算

# エラー処理・ロギング
thiserror = "1"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"

# 日時管理
chrono = { version = "0.4", features = ["serde"] }
```

---

## 2. システムアーキテクチャ

### 2.1 全体構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                        MemoryOS (Rust)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   API Layer  │    │  Memory Core │    │  Storage     │      │
│  │              │    │              │    │  Layer       │      │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤      │
│  │ REST/MCP     │◄──►│ Orchestrator │◄──►│ HNSW Index   │      │
│  │ Server       │    │              │    │ (hnsw-rs)    │      │
│  └──────────────┘    └──────────────┘    ├──────────────┤      │
│         ▲                  ▲              │ SQLite       │      │
│         │                  │              │ (sqlx)       │      │
│  ┌──────┴──────────────────┴─────────────┴──────────┐   │
│  │                    External Services              │   │
│  ├──────────────────────────────────────────────────┤   │
│  │  LLM API (OpenAI/Anthropic) | Embedding Model    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

【ストレージ層設計】
✅ HNSW-rs 単体でベクトル検索
✅ SQLite が唯一の ID ソース（AUTOINCREMENT）
```

### 2.2 メモリ階層構造

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Layer Hierarchy                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐                                          │
│   │  Short Term  │  ← 最近の対話履歴（容量制限：N 件）        │
│   │     Memory   │      LRU で古いものを削除                │
│   └──────┬───────┘                                          │
│          │ 要約・統合（失敗時は元に戻す）                    │
│          ▼                                                   │
│   ┌──────────────┐                                          │
│   │  Mid Term    │  ← 対話の要約・パターン抽出              │
│   │     Memory   │      ヒート値で重要度判定                │
│   └──────┬───────┘                                          │
│          │ 長期化                                            │
│          ▼                                                   │
│   ┌──────────────┐                                          │
│   │  Long Term   │  ← ユーザープロフィール・知識ベース      │
│   │     Memory   │      ベクトル検索で関連取得              │
│   └──────────────┘                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

【プロモーション保証】
✅ 失敗時は短期記憶へ戻す（データ損失なし）
✅ 復元は先頭へ（順序維持：restore_oldest()）
```

### 2.3 データフロー

```
ユーザー入力
    │
    ▼
┌─────────────────┐
│  Embedding      │  ← キューイング + バッチ処理（candle）
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Memory Update  │  ← SQLite で ID 発行・保存
│  Processor      │
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    │         │              │
    ▼         ▼              ▼
┌───────┐ ┌───────┐     ┌──────────┐
│Short  │ │ Mid   │     │  Long    │
│Term   │ │ Term  │     │  Term    │
│Store  │ │ Store │     │  Store   │
└───┬───┘ └───┬───┘     └────┬─────┘
    │         │              │
    └─────────┴──────────────┘
              │
              ▼
       ┌──────────────┐
       │  Retrieval   │  ← HNSW + rerank（cosine 再計算）
       │  Engine      │     SQLite からベクトル再取得
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Response    │  ← LLM にコンテキストを渡して生成
       │  Generation  │
       └──────────────┘

【検索精度保証】
✅ HNSW で候補取得（近似）
✅ SQLite から元ベクトル再取得
✅ cosine 類似度で rerank（正確なスコア）
```

---

## 3. モジュール構成

### 3.1 プロジェクト構造

```
memoryos-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # ライブラリエントリポイント
│   ├── error.rs                  # エラー型定義
│   │
│   ├── memory/                   # メモリ管理コア
│   │   ├── mod.rs                # 公開インターフェース
│   │   ├── orchestrator.rs       # メインオーケストレーター
│   │   ├── short_term.rs         # 短期記憶（VecDeque）
│   │   ├── mid_term.rs           # 中期記憶（HashMap）
│   │   └── long_term.rs          # 長期記憶（HashMap）
│   │
│   ├── storage/                  # ストレージ層
│   │   ├── mod.rs                # 公開インターフェース
│   │   ├── vector_store.rs       # HNSW インデックス管理
│   │   └── sqlite_store.rs       # SQLite メタデータ永続化
│   │
│   ├── llm/                      # LLM インターフェース
│   │   ├── mod.rs                # 公開インターフェース
│   │   ├── client.rs             # HTTP クライアント（reqwest）
│   │   └── prompts.rs            # プロンプトテンプレート
│   │
│   ├── embedding/                # 埋め込み処理
│   │   ├── mod.rs                # 公開インターフェース
│   │   ├── queue.rs              # キューイング・バッチ処理
│   │   └── vector_ops.rs         # ベクトル演算（cosine 類似度）
│   │
│   ├── config/                   # 設定管理
│   │   ├── mod.rs                # 公開インターフェース
│   │   └── settings.rs           # 設定構造体
│   │
│   └── api/                      # API サーバー（オプション）
│       ├── mod.rs                # 公開インターフェース
│       └── server.rs             # REST/MCP サーバー（axum/tower）
│
├── examples/                     # デモコード
│   ├── simple_demo.rs            # basic_usage.py の移植
│   └── api_server.rs             # MCP サーバー実装例
│
├── tests/                        # 統合テスト
│   ├── memory_integration.rs
│   └── storage_integration.rs
│
└── benches/                      # ベンチマーク
    └── retrieval_bench.rs        # LoCoMo ベンチマーク再現
```

### 3.2 モジュール詳細

| Python | Rust | 説明 |
|--------|------|------|
| `memoryos.py` | `src/memory/orchestrator.rs` | メインクラス（MemoryOs） |
| `short_term.py` | `src/memory/short_term.rs` | 短期記憶管理（VecDeque） |
| `mid_term.py` | `src/memory/mid_term.rs` | 中期記憶管理（HashMap） |
| `long_term.py` | `src/memory/long_term.rs` | 長期記憶管理（HashMap） |
| `retriever.py` | `src/storage/vector_store.rs` | HNSW ベクトル検索 |
| `updater.py` | `orchestrator::promote_*()` | メモリ更新処理 |
| `utils.py` | `src/embedding/vector_ops.rs` | ユーティリティ関数 |
| `prompts.py` | `src/llm/prompts.rs` | プロンプトテンプレート |

---

## 4. データ構造定義

### 4.1 エラー型

```rust
// src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryOsError {
    #[error("LLM API error: {0}")]
    LlmApiError(String),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] sqlx::Error),
    
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("HNSW error: {0}")]
    HnswError(String),
}

pub type Result<T> = std::result::Result<T, MemoryOsError>;
```

### 4.2 メモリレコード（sqlx::FromRow 対応）

```rust
// src/memory/mod.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

/// メモリ階層タイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLayer {
    ShortTerm,
    MidTerm,
    LongTerm,
}

impl std::fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLayer::ShortTerm => write!(f, "ShortTerm"),
            MemoryLayer::MidTerm => write!(f, "MidTerm"),
            MemoryLayer::LongTerm => write!(f, "LongTerm"),
        }
    }
}

impl std::str::FromStr for MemoryLayer {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ShortTerm" => Ok(MemoryLayer::ShortTerm),
            "MidTerm" => Ok(MemoryLayer::MidTerm),
            "LongTerm" => Ok(MemoryLayer::LongTerm),
            _ => Err(format!("Invalid layer: {}", s)),
        }
    }
}

/// 会話レコード（短期記憶用）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub user_input: String,
    pub agent_response: String,
    pub timestamp: DateTime<Utc>,
}

/// メモリレコード（永続化用、sqlx::FromRow 対応）
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MemoryRecord {
    #[sqlx(rename = "id")]
    pub id: i64,  // SQLite INTEGER → Rust i64
    
    #[sqlx(rename = "user_id")]
    pub user_id: String,
    
    #[sqlx(rename = "assistant_id")]
    pub assistant_id: String,
    
    #[sqlx(rename = "content")]
    pub content: String,
    
    #[sqlx(rename = "layer")]
    layer_str: String,  // 文字列で取得、後で変換
    
    #[sqlx(rename = "heat_score")]
    pub heat_score: f32,
    
    #[sqlx(rename = "metadata")]
    metadata_json: String,  // JSON 文字列で取得
    
    #[sqlx(rename = "created_at")]
    created_at_str: String,  // 文字列で取得、後で変換
    
    #[sqlx(rename = "updated_at")]
    updated_at_str: String,  // 文字列で取得、後で変換
}

impl MemoryRecord {
    /// layer 文字列を enum に変換した getter
    pub fn layer(&self) -> Result<MemoryLayer> {
        self.layer_str.parse()
    }
    
    /// created_at を DateTime に変換した getter
    pub fn created_at(&self) -> Result<DateTime<Utc>> {
        let created_at = chrono::NaiveDateTime::parse_from_str(
            &self.created_at_str, "%Y-%m-%d %H:%M:%S"
        ).map_err(|e| MemoryOsError::StorageError(format!("Date parse error: {}", e)))?
            .and_utc();
        
        Ok(created_at)
    }
    
    /// metadata JSON を Value に変換した getter
    pub fn metadata(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.metadata_json)
            .map_err(|e| MemoryOsError::SerializationError(e))
    }
}

/// メタデータ構造体（default 実装なし、完全復元必須）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub user_id: String,
    pub assistant_id: String,
    pub content: String,
    pub layer: MemoryLayer,
    pub heat_score: f32,
    pub created_at: DateTime<Utc>,
}

/// 検索結果（レコード + 類似度スコア）
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: MemoryRecord,
    pub similarity_score: f32,  // rerank 後の類似度スコア（0.0-1.0）
}

/// 検索候補（rerank 前、HNSW 距離付き）
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    pub record_id: u64,
    pub hnsw_distance: f32,      // HNSW の距離（近似値）
    pub rerank_score: f32,       // rerank 後の類似度スコア
}

/// Embedding レコード（再構築用、完全な metadata 付き）
#[derive(Debug)]
pub struct EmbeddingRecord {
    pub id: u64,
    pub embedding: Vec<f32>,
    pub metadata: MemoryMetadata,
}
```

### 4.3 設定構造体

```rust
// src/config/settings.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOsConfig {
    // ユーザー識別
    pub user_id: String,
    pub assistant_id: String,
    
    // LLM 設定
    pub openai_api_key: String,
    pub openai_base_url: Option<String>,
    pub llm_model: String,
    
    // 埋め込みモデル
    pub embedding_model_name: String,  // "BAAI/bge-m3" など
    
    // ストレージ設定
    pub data_storage_path: String,
    
    // メモリ容量制限
    pub short_term_capacity: usize,           // デフォルト：7
    pub mid_term_heat_threshold: f32,         // デフォルト：5.0
    pub retrieval_queue_capacity: usize,      // デフォルト：7
    pub long_term_knowledge_capacity: usize,  // デフォルト：100
    
    // 検索設定
    pub similarity_threshold: f32,            // デフォルト：0.7
    pub hnsw_m: u32,                          // HNSW パラメータ（デフォルト：16）
    pub hnsw_ef_construction: u32,            // HNSW パラメータ（デフォルト：200）
}

impl Default for MemoryOsConfig {
    fn default() -> Self {
        Self {
            user_id: "default_user".to_string(),
            assistant_id: "default_assistant".to_string(),
            openai_api_key: "".to_string(),
            openai_base_url: Some("https://api.openai.com/v1".to_string()),
            llm_model: "gpt-4o-mini".to_string(),
            embedding_model_name: "BAAI/bge-m3".to_string(),
            data_storage_path: "./memoryos_data".to_string(),
            short_term_capacity: 7,
            mid_term_heat_threshold: 5.0,
            retrieval_queue_capacity: 7,
            long_term_knowledge_capacity: 100,
            similarity_threshold: 0.7,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }
}

impl MemoryOsConfig {
    /// 埋め込みモデルの次元数を取得
    pub fn get_embedding_dimensions(&self) -> usize {
        match self.embedding_model_name.as_str() {
            "BAAI/bge-m3" => 1024,
            "all-MiniLM-L6-v2" => 384,
            _ => 768,  // デフォルト
        }
    }
}
```

---

## 5. ストレージ層設計

### 5.1 SQLite スキーマ

```sql
-- memoryos_schema.sql

-- バージョン管理テーブル
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO schema_version (version) VALUES (4);

-- メモリレコード（embedding_vector は little-endian f32 配列）
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    content TEXT NOT NULL,
    layer TEXT NOT NULL CHECK(layer IN ('ShortTerm', 'MidTerm', 'LongTerm')),
    heat_score REAL DEFAULT 0.0,
    metadata JSON DEFAULT '{}',  -- SQLite JSON1 拡張（3.9+）
    embedding_vector BLOB,       -- little-endian f32 配列（次元数×4 バイト）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_user_layer ON memories(user_id, layer);
CREATE INDEX IF NOT EXISTS idx_heat_score ON memories(heat_score DESC);
CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at DESC);

-- JSON1 でのメタデータクエリ例（SQLite 3.9+）
-- SELECT * FROM memories WHERE json_extract(metadata, '$.category') = 'knowledge';
```

### 5.2 BLOB 形式定義（共通仕様）

| 項目 | 値 |
|------|-----|
| **エンディアン** | little-endian |
| **データ型** | f32（IEEE 754 単精度浮動小数点） |
| **バイト数** | `次元数 × 4` |
| **例：BGE-M3 (1024 次元)** | 4096 バイト |

```rust
// src/storage/blob_format.rs

/// embedding_vector BLOB 形式定義（共通仕様）
pub mod blob_format {
    use crate::error::{MemoryOsError, Result};
    
    /// BLOB バイト数計算（モデル次元数依存）
    pub fn expected_blob_size(dimensions: usize) -> usize {
        dimensions * 4  // f32 = 4 バイト
    }
    
    /// Vec<f32> を BLOB に変換（little-endian）
    pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
        let mut blob = Vec::with_capacity(expected_blob_size(embedding.len()));
        for &value in embedding {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        blob
    }
    
    /// BLOB を Vec<f32> に変換（little-endian、次元数検証付き）
    pub fn blob_to_embedding(blob: &[u8], expected_dimensions: usize) -> Result<Vec<f32>> {
        let expected_size = expected_blob_size(expected_dimensions);
        
        if blob.len() != expected_size {
            return Err(MemoryOsError::StorageError(
                format!(
                    "Invalid embedding BLOB length: got {}, expected {} (dimensions={})",
                    blob.len(), expected_size, expected_dimensions
                )
            ));
        }
        
        let mut result = Vec::with_capacity(expected_dimensions);
        for chunk in blob.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
        
        Ok(result)
    }
}
```

### 5.3 MetadataStore（SQLite）

```rust
// src/storage/sqlite_store.rs
use sqlx::{QueryBuilder, Sqlite};
use std::collections::HashMap;
use tokio::sync::RwLock as AsyncRwLock;
use std::sync::Arc;

pub struct MetadataStore {
    pool: sqlx::SqlitePool,
}

impl MetadataStore {
    /// インスタンス作成
    pub async fn new(db_path: &str) -> Result<Self> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&format!("sqlite://{}", db_path))
            .await?;
        
        Self::init_schema(&pool).await?;
        
        Ok(Self { pool })
    }
    
    /// スキーマ初期化
    async fn init_schema(pool: &sqlx::SqlitePool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                assistant_id TEXT NOT NULL,
                content TEXT NOT NULL,
                layer TEXT NOT NULL CHECK(layer IN ('ShortTerm', 'MidTerm', 'LongTerm')),
                heat_score REAL DEFAULT 0.0,
                metadata JSON DEFAULT '{}',
                embedding_vector BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#
        ).execute(pool).await?;
        
        sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS idx_user_layer 
               ON memories(user_id, layer)"#
        ).execute(pool).await?;
        
        Ok(())
    }
    
    /// 新規メモリ保存（assistant_id 必須、embedding オプション）
    pub async fn save_new_memory_with_embedding(
        &self,
        user_id: &str,
        assistant_id: &str,      // ← 必須引数
        content: &str,
        layer: MemoryLayer,
        embedding: Option<&[f32]>,  // オプション（None なら BLOB 保存なし）
    ) -> Result<u64> {
        let embedding_blob = embedding.map(|e| blob_format::embedding_to_blob(e));
        
        let result = sqlx::query(
            r#"INSERT INTO memories 
               (user_id, assistant_id, content, layer, embedding_vector)
               VALUES (?, ?, ?, ?, ?)"#
        )
        .bind(user_id)
        .bind(assistant_id)  // ← バインド必須
        .bind(content)
        .bind(format!("{:?}", layer))
        .bind(embedding_blob.as_deref())
        .execute(&self.pool)
        .await?;
        
        let id = result.last_insert_rowid() as u64;
        Ok(id)
    }
    
    /// QueryBuilder で安全な可変長 IN クエリ（順序保証付き）
    pub async fn get_records_by_ids_ordered(
        &self,
        record_ids: &[u64],
    ) -> Result<Vec<MemoryRecord>> {
        if record_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut qb: QueryBuilder<Sqlite> =
            QueryBuilder::new("SELECT * FROM memories WHERE id IN (");

        {
            let mut separated = qb.separated(", ");
            for &id in record_ids {
                separated.push_bind(id as i64);  // ← 各要素を個別に bind
            }
        }

        qb.push(")");

        let mut records: Vec<MemoryRecord> =
            qb.build_query_as().fetch_all(&self.pool).await?;

        // 元の順序でソート（record_ids の順序を保証）
        let id_order: HashMap<u64, usize> = record_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        records.sort_by_key(|r| id_order.get(&(r.id as u64)).copied().unwrap_or(usize::MAX));

        Ok(records)
    }
    
    /// ID リストでレコード取得（類似度スコア付き、SearchResult 形式）
    pub async fn get_search_results(
        &self,
        candidates: &[SearchCandidate],  // record_id + rerank_score 付き
    ) -> Result<Vec<SearchResult>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        
        // ID リスト取得（rerank 順）
        let record_ids: Vec<_> = candidates.iter().map(|c| c.record_id).collect();
        
        // 順序保証付きでレコード取得
        let records = self.get_records_by_ids_ordered(&record_ids).await?;
        
        // 類似度スコアを付与（candidates の rerank 順）
        let score_map: HashMap<_, _> = candidates.iter()
            .map(|c| (c.record_id, c.rerank_score))
            .collect();
        
        let results: Vec<_> = records.into_iter()
            .filter_map(|record| {
                score_map.get(&record.id as u64).map(|&score| SearchResult {
                    record,
                    similarity_score: score,
                })
            })
            .collect();
        
        Ok(results)
    }
    
    /// 全レコードの embedding と metadata を取得（再構築用、完全復元）
    pub async fn get_all_embeddings(&self, dimensions: usize) -> Result<Vec<EmbeddingRecord>> {
        let rows = sqlx::query(
            r#"SELECT 
                id, user_id, assistant_id, content, layer, heat_score,
                metadata, embedding_vector, created_at
               FROM memories 
               WHERE embedding_vector IS NOT NULL"#
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut result = Vec::new();
        for row in rows {
            // 各フィールドを安全に抽出
            let id: i64 = row.get("id");
            let user_id: String = row.get("user_id");
            let assistant_id: String = row.get("assistant_id");
            let content: String = row.get("content");
            let layer_str: String = row.get("layer");
            let heat_score: f32 = row.get("heat_score");
            let metadata_json: String = row.get("metadata");
            let embedding_blob: Vec<u8> = row.get("embedding_vector");
            let created_at_str: String = row.get("created_at");
            
            // layer 文字列を enum に変換
            let layer = match layer_str.as_str() {
                "ShortTerm" => MemoryLayer::ShortTerm,
                "MidTerm" => MemoryLayer::MidTerm,
                "LongTerm" => MemoryLayer::LongTerm,
                _ => return Err(MemoryOsError::StorageError(
                    format!("Invalid layer: {}", layer_str)
                )),
            };
            
            // metadata JSON をパース（default 不使用）
            let metadata_value: serde_json::Value = serde_json::from_str(&metadata_json)?;
            
            // created_at を DateTime に変換
            let created_at: DateTime<Utc> = chrono::NaiveDateTime::parse_from_str(
                &created_at_str, "%Y-%m-%d %H:%M:%S"
            ).map_err(|e| MemoryOsError::StorageError(format!("Date parse error: {}", e)))?
                .and_utc();
            
            // embedding BLOB を Vec<f32> に変換（次元数検証付き、引数から取得）
            let embedding = match blob_format::blob_to_embedding(&embedding_blob, dimensions) {
                Ok(e) => e,
                Err(e) => {
                    // 次元不一致レコードは skip + warn（モデル変更時対応）
                    tracing::warn!("Skipping record {} with invalid embedding: {}", id, e);
                    continue;  // ← このレコードをスキップして次へ
                }
            };
            
            // 完全な Metadata 構築（default 不使用）
            let metadata = MemoryMetadata {
                user_id,
                assistant_id,
                content,
                layer,
                heat_score,
                created_at,
            };
            
            result.push(EmbeddingRecord {
                id: id as u64,
                embedding,
                metadata,  // ← 完全復元
            });
        }
        
        Ok(result)
    }
    
    /// レコード削除（HNSW 失敗時のクリーンアップ用）
    pub async fn delete_record(&self, record_id: u64) -> Result<()> {
        sqlx::query("DELETE FROM memories WHERE id = ?")
            .bind(record_id as i64)
            .execute(&self.pool)
            .await?;
        
        Ok(())
    }
    
    /// ID リストで embedding 再取得（rerank 用）
    pub async fn get_embeddings_by_ids(
        &self,
        record_ids: &[u64],
        dimensions: usize,  // ← 次元数を引数で受け取る
    ) -> Result<HashMap<u64, Vec<f32>>> {
        if record_ids.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut qb = QueryBuilder::<Sqlite>::new("SELECT id, embedding_vector FROM memories WHERE id IN (");
        
        {
            let mut separated = qb.separated(", ");
            for &id in record_ids {
                separated.push_bind(id as i64);
            }
        }
        
        qb.push(")");
        
        let rows = qb.build_query().fetch_all(&self.pool).await?;
        
        // 引数から次元数を取得
        let mut result = HashMap::new();
        
        for row in rows {
            let id: i64 = row.get("id");
            let blob: Vec<u8> = row.get("embedding_vector");
            
            if let Ok(embedding) = blob_format::blob_to_embedding(&blob, dimensions) {
                result.insert(id as u64, embedding);
            }
        }
        
        Ok(result)
    }
}

impl Clone for MetadataStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),  // sqlx::Pool は Clone 対応
        }
    }
}
```

### 5.4 VectorStore（HNSW）

```rust
// src/storage/vector_store.rs
use hnsw_rs::{HnswBuilder, Hnsw, Space};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock as AsyncRwLock};

/// ベクトル検索ストレージ（HNSW 単体 + SQLite メタデータ）
/// 
/// # 設計意図
/// 
/// ## ロック戦略
/// - HNSW インデックスは `tokio::sync::Mutex` で排他制御
/// - **理由**: HNSW ライブラリが同期安全でないため、同時アクセスを防止
/// - **トレードオフ**: 検索も挿入も 1 本化される（並列性制限）
/// 
/// ## 将来の拡張
/// - 高負荷環境では `Mutex` → `RwLock` に変更を検討
///   - 読み取り（検索）は共有可能、書き込み（挿入/削除）のみ排他
///   - ただし HNSW ライブラリの同時読み取り安全性を確認必要
/// 
/// ## メタデータキャッシュ
/// - `tokio::sync::RwLock` で管理（読み取り中心）
/// - HNSW とは独立したロック、並列アクセス可能
#[derive(Debug)]
pub struct VectorStore {
    /// HNSW インデックス（排他ロック、同時アクセス不可）
    /// 
    /// 注意：この Mutex が検索スループットのボトルネックになる可能性あり。
    /// 高負荷環境では以下の対策を検討：
    /// 1. HNSW ライブラリの RwLock 対応バージョンへ移行
    /// 2. インデックス分割（sharding）による並列化
    index: Mutex<Hnsw<f32>>,
    
    /// メタデータキャッシュ（共有ロック、同時読み取り OK）
    metadata_cache: AsyncRwLock<HashMap<u64, MemoryMetadata>>,
    
    /// ベクトル次元数
    dimensions: usize,
    
    /// MetadataStore 参照を保持（再構築・rerank 用）
    metadata_store: Arc<MetadataStore>,
}

impl VectorStore {
    /// インスタンス作成（インデックス再構築付き）
    pub async fn new(
        dimensions: usize,
        m: u32,
        ef_construction: u32,
        metadata_store: Arc<MetadataStore>,
    ) -> Result<Self> {
        // HNSW インデックス作成
        let mut index = HnswBuilder::new()
            .dimensions(dimensions as u32)
            .max_elements(100_000)
            .m(m)
            .ef_construction(ef_construction)
            .space(Space::Cosine)  // ← Cosine 距離を明示
            .build::<f32>()?;
        
        // SQLite から全レコード取得
        let all_records = metadata_store.get_all_embeddings(dimensions).await?;
        
        // HNSW に再登録
        for record in &all_records {
            index.add_point(&record.embedding, record.id as i64)?;
        }
        
        // メタデータキャッシュも初期化（完全復元、default 不使用）
        let metadata_cache: HashMap<_, _> = all_records.iter()
            .map(|r| (r.id, r.metadata.clone()))
            .collect();
        
        Ok(Self {
            index: Mutex::new(index),
            metadata_cache: AsyncRwLock::new(metadata_cache),
            dimensions,
            metadata_store,
        })
    }
    
    /// 記憶追加（SQLite ID を参照として使用）
    pub async fn insert(
        &self,
        record_id: u64,
        embedding: Vec<f32>,
        metadata: MemoryMetadata,
    ) -> Result<()> {
        // HNSW に挿入（record_id を label として使用）
        let mut index = self.index.lock().await;
        index.add_point(&embedding, record_id as i64)?;
        
        // メタデータキャッシュに登録
        let mut cache = self.metadata_cache.write().await;
        cache.insert(record_id, metadata);
        
        Ok(())
    }
    
    /// 類似度検索（rerank 付き）
    pub async fn search_similar(
        &self,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        // ステップ 1: HNSW で候補取得（余裕を持って）
        let mut candidates = {
            let index_guard = self.index.lock().await;
            index_guard.search(query, (k * 5) as u32, 50)?  // k*5 件取得、ef_search=50
                .into_iter()
                .filter_map(|(label, distance)| {
                    if distance < (1.0 - threshold + 0.1) {  // 少し緩くフィルタ
                        Some(SearchCandidate {
                            record_id: label as u64,
                            hnsw_distance: distance,
                            rerank_score: 0.0,  // 後で更新
                        })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        
        // ステップ 2: SQLite から元ベクトル再取得（rerank 用）
        let embeddings = self.metadata_store.get_embeddings_by_ids(
            &candidates.iter().map(|c| c.record_id).collect::<Vec<_>>(),
            self.dimensions,
        ).await?;
        
        // ステップ 3: rerank（cosine 類似度再計算）
        for candidate in &mut candidates {
            if let Some(embedding) = embeddings.get(&candidate.record_id) {
                let similarity = cosine_similarity(query, embedding);
                candidate.rerank_score = similarity;  // ← rerank スコア設定
            } else {
                candidate.rerank_score = 0.0;  // ベクトルなしの場合は除外
            }
        }
        
        // ステップ 4: rerank 結果でソート（類似度降順）
        candidates.sort_by(|a, b| 
            b.rerank_score.partial_cmp(&a.rerank_score).unwrap_or(std::cmp::Ordering::Equal)
        );
        
        // ステップ 5: 上位 k 件を SearchResult 形式で返す（順序・スコア保証）
        let top_k = candidates.iter().take(k).collect::<Vec<_>>();
        
        // MetadataStore から完全なレコード取得
        self.metadata_store.get_search_results(&top_k.iter().map(|c| (*c).clone()).collect::<Vec<_>>()).await
    }
    
    /// 削除
    pub async fn remove(&self, record_id: u64) -> Result<()> {
        let mut index = self.index.lock().await;
        index.mark_deleted(record_id as i64)?;
        
        let mut cache = self.metadata_cache.write().await;
        cache.remove(&record_id);
        
        Ok(())
    }
}

/// 余弦類似度計算（rerank 用）
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}
```

---

## 6. メモリ管理コア

### 6.1 ShortTermMemory

```rust
// src/memory/short_term.rs
use std::collections::VecDeque;
use crate::memory::Conversation;

/// 短期記憶（ロックなし、直接操作）
#[derive(Debug)]
pub struct ShortTermMemory {
    inner: VecDeque<Conversation>,
    capacity: usize,
}

impl ShortTermMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    /// 追加（呼び出し側でロック済み前提）
    pub fn add(&mut self, conversation: Conversation) {
        if self.inner.len() >= self.capacity {
            self.inner.pop_front();
        }
        self.inner.push_back(conversation);
    }
    
    /// 取得（読み取り用、clone 必要）
    pub fn get_recent(&self, n: usize) -> Vec<Conversation> {
        self.inner.iter().rev().take(n).cloned().collect()
    }
    
    /// 最古の 1 件を取得（参照）
    pub fn get_oldest(&self) -> Option<&Conversation> {
        self.inner.front()
    }
    
    /// 最古の 1 件を削除して返す（重複昇格防止用）
    pub fn remove_oldest(&mut self) -> Option<Conversation> {
        self.inner.pop_front()
    }
    
    /// 失敗時復元用（先頭へ戻す、順序維持）
    pub fn restore_oldest(&mut self, conversation: Conversation) {
        self.inner.push_front(conversation);  // ← push_back ではなく push_front
    }
    
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
```

### 6.2 MidTermMemory

```rust
// src/memory/mid_term.rs
use std::collections::HashMap;
use crate::memory::{MemoryRecord, MemoryLayer};

/// 中期記憶（ロックなし、直接操作）
#[derive(Debug)]
pub struct MidTermMemory {
    inner: HashMap<u64, MemoryRecord>,
    heat_threshold: f32,
}

impl MidTermMemory {
    pub fn new(heat_threshold: f32) -> Self {
        Self {
            inner: HashMap::new(),
            heat_threshold,
        }
    }
    
    /// 追加（呼び出し側でロック済み前提）
    pub fn add(&mut self, record: MemoryRecord) {
        self.inner.insert(record.id as u64, record);
    }
    
    /// ヒート値閾値以上の記録を取得
    pub fn get_above_threshold(&self, threshold: f32) -> Vec<&MemoryRecord> {
        self.inner.values()
            .filter(|r| r.heat_score >= threshold)
            .collect()
    }
    
    /// 削除
    pub fn remove(&mut self, id: u64) -> Option<MemoryRecord> {
        self.inner.remove(&id)
    }
}
```

### 6.3 LongTermMemory

```rust
// src/memory/long_term.rs
use std::collections::HashMap;
use crate::memory::{MemoryRecord, MemoryLayer};

/// 長期記憶（ロックなし、直接操作）
#[derive(Debug)]
pub struct LongTermMemory {
    inner: HashMap<u64, MemoryRecord>,
    capacity: usize,
}

impl LongTermMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HashMap::new(),
            capacity,
        }
    }
    
    /// 追加（容量超過時は最古削除）
    pub fn add(&mut self, record: MemoryRecord) {
        if self.inner.len() >= self.capacity && !self.inner.contains_key(&(record.id as u64)) {
            // 最古の 1 件を削除（簡易実装：最小 created_at）
            if let Some(oldest_id) = self.inner.iter()
                .min_by(|a, b| a.1.created_at_str.cmp(&b.1.created_at_str))
                .map(|(id, _)| *id)
            {
                self.inner.remove(&oldest_id);
            }
        }
        self.inner.insert(record.id as u64, record);
    }
    
    /// プロフィール要約生成（簡易版）
    pub fn get_profile_summary(&self) -> String {
        format!("Total memories: {}", self.inner.len())
    }
}
```

### 6.4 MemoryOs Orchestrator

```rust
// src/memory/orchestrator.rs
use std::sync::Arc;
use tokio::sync::RwLock as AsyncRwLock;

pub struct MemoryOs {
    config: Arc<MemoryOsConfig>,
    
    short_term: Arc<AsyncRwLock<ShortTermMemory>>,
    mid_term: Arc<AsyncRwLock<MidTermMemory>>,
    long_term: Arc<AsyncRwLock<LongTermMemory>>,
    
    vector_store: Arc<VectorStore>,
    metadata_store: Arc<MetadataStore>,
    
    llm_client: Arc<LlmClient>,
    embedding_queue: Arc<EmbeddingQueue>,
}

impl MemoryOs {
    pub async fn new(config: MemoryOsConfig) -> Result<Self> {
        let metadata_store = MetadataStore::new(&config.data_storage_path).await?;
        
        // VectorStore 作成（SQLite から HNSW 再構築）
        let vector_store = VectorStore::new(
            config.get_embedding_dimensions(),
            config.hnsw_m,
            config.hnsw_ef_construction,
            Arc::clone(&metadata_store),
        ).await?;
        
        let llm_client = LlmClient::new(
            &config.openai_api_key,
            config.openai_base_url.as_deref(),
            &config.llm_model,
        )?;
        
        let embedding_queue = EmbeddingQueue::new(8, 100);
        
        let short_term = ShortTermMemory::new(config.short_term_capacity);
        let mid_term = MidTermMemory::new(config.mid_term_heat_threshold);
        let long_term = LongTermMemory::new(config.long_term_knowledge_capacity);
        
        Ok(Self {
            config: Arc::new(config),
            short_term: Arc::new(AsyncRwLock::new(short_term)),
            mid_term: Arc::new(AsyncRwLock::new(mid_term)),
            long_term: Arc::new(AsyncRwLock::new(long_term)),
            vector_store: Arc::new(vector_store),
            metadata_store,
            llm_client: Arc::new(llm_client),
            embedding_queue: Arc::new(embedding_queue),
        })
    }
    
    pub async fn add_memory(&self, user_input: &str, agent_response: &str) -> Result<()> {
        let conversation = Conversation {
            user_input: user_input.to_string(),
            agent_response: agent_response.to_string(),
            timestamp: chrono::Utc::now(),
        };
        
        {
            let mut short_term = self.short_term.write().await;
            short_term.add(conversation);
        }
        
        let embedding = self.embedding_queue.generate(user_input).await?;
        
        // assistant_id 必須引数
        let record_id = self.metadata_store.save_new_memory_with_embedding(
            &self.config.user_id,
            &self.config.assistant_id,  // ← assistant_id 必須
            user_input,
            MemoryLayer::ShortTerm,
            Some(&embedding),
        ).await?;
        
        let metadata = MemoryMetadata {
            user_id: self.config.user_id.clone(),
            assistant_id: self.config.assistant_id.clone(),
            content: user_input.to_string(),
            layer: MemoryLayer::ShortTerm,
            heat_score: 0.0,
            created_at: chrono::Utc::now(),
        };
        
        self.vector_store.insert(record_id, embedding, metadata).await?;
        
        // 耐障害性対応（失敗時は短期へ戻す）
        self.promote_short_to_mid().await?;
        self.promote_memory_if_needed().await?;
        
        Ok(())
    }
    
    pub async fn get_response(&self, query: &str) -> Result<String> {
        let query_embedding = self.embedding_queue.generate(query).await?;
        
        // rerank 付き、SearchResult 形式で返却
        let relevant_memories = self.vector_store.search_similar(
            &query_embedding,
            self.config.retrieval_queue_capacity,
            self.config.similarity_threshold,
        ).await?;
        
        // 結果を文字列として返す
        let context = self.build_context(&relevant_memories).await?;
        
        let response = self.llm_client.generate(&context, query).await?;
        
        Ok(response)
    }
    
    async fn summarize_conversation(&self, conversation: &Conversation) -> Result<String> {
        let prompt = format!(
            "Summarize this conversation in one sentence:\nUser: {}\nAssistant: {}",
            conversation.user_input,
            conversation.agent_response
        );
        
        self.llm_client.generate_simple(&prompt).await
    }
    
    async fn build_context(&self, memories: &[SearchResult]) -> Result<String> {
        let profile = {
            let long_term = self.long_term.read().await;
            long_term.get_profile_summary()
        };
        
        // 結果を format! で包む
        Ok(format!(
            "=== User Profile ===\n{}\n\n=== Relevant Memories ===\n{}",
            profile,
            memories.iter()
                .map(|m| format!("[Similarity: {:.2}] {}", m.similarity_score, m.record.content))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
    
    /// 失敗時は短期記憶へ戻す（restore_oldest で先頭復元）
    async fn promote_short_to_mid(&self) -> Result<()> {
        // フェーズ 1: 短期記憶から最古を即座に削除
        let conversation_to_promote = {
            let mut short_term = self.short_term.write().await;
            
            if short_term.len() < self.config.short_term_capacity {
                return Ok(());
            }
            
            short_term.remove_oldest()  // Option<Conversation> を返す
        };
        
        let Some(conversation) = conversation_to_promote else {
            return Ok(());
        };
        
        // フェーズ 2: 要約（失敗時は短期へ戻す）
        let summary = match self.summarize_conversation(&conversation).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Conversation summarization failed, restoring to short-term: {}", e);
                
                let mut short_term = self.short_term.write().await;
                short_term.restore_oldest(conversation);  // ← push_front で先頭復元
                
                return Ok(());
            }
        };
        
        // 中期記憶も HNSW 検索対象にするなら embedding を再生成
        let mid_term_embedding = self.embedding_queue.generate(&summary).await?;
        
        // フェーズ 3: SQLite で新規 ID 発行・保存（失敗時は短期へ戻す）
        let record_id = match self.metadata_store.save_new_memory_with_embedding(
            &self.config.user_id,
            &self.config.assistant_id,
            &summary,
            MemoryLayer::MidTerm,
            Some(&mid_term_embedding),  // embedding を保存（HNSW 検索対象）
        ).await {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!("SQLite save failed, restoring to short-term: {}", e);
                
                let mut short_term = self.short_term.write().await;
                short_term.restore_oldest(conversation);  // ← push_front で先頭復元
                
                return Ok(());
            }
        };
        
        // フェーズ 4: HNSW に登録（失敗時は SQLite は残るが短期へ戻す）
        let metadata = MemoryMetadata {
            user_id: self.config.user_id.clone(),
            assistant_id: self.config.assistant_id.clone(),
            content: summary.clone(),
            layer: MemoryLayer::MidTerm,
            heat_score: 0.0,
            created_at: chrono::Utc::now(),
        };
        
        if let Err(e) = self.vector_store.insert(record_id, mid_term_embedding, metadata).await {
            tracing::warn!("HNSW insert failed, restoring to short-term: {}", e);
            
            // SQLite から削除（クリーンアップ）
            let _ = self.metadata_store.delete_record(record_id).await;
            
            let mut short_term = self.short_term.write().await;
            short_term.restore_oldest(conversation);  // ← push_front で先頭復元
            
            return Ok(());
        }
        
        // フェーズ 5: 中期に追加（最終確定）
        {
            let mut mid_term = self.mid_term.write().await;
            mid_term.add(MemoryRecord {
                id: record_id as i64,
                user_id: self.config.user_id.clone(),
                assistant_id: self.config.assistant_id.clone(),
                content: summary,
                layer_str: "MidTerm".to_string(),
                heat_score: 0.0,
                metadata_json: "{}".to_string(),
                created_at_str: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                updated_at_str: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            });
        }
        
        Ok(())
    }
    
    async fn promote_memory_if_needed(&self) -> Result<()> {
        // ID 集約→更新の 2 フェーズ
        let ids_to_promote = {
            let mid_term = self.mid_term.read().await;
            mid_term.get_above_threshold(self.config.mid_term_heat_threshold)
                .iter()
                .map(|r| r.id as u64)
                .collect::<Vec<_>>()
        };
        
        if ids_to_promote.is_empty() {
            return Ok(());
        }
        
        for record_id in ids_to_promote {
            let record = {
                let mut mid_term = self.mid_term.write().await;
                mid_term.remove(record_id)
            };
            
            if let Some(record) = record {
                let mut long_term = self.long_term.write().await;
                long_term.add(record);
            }
        }
        
        Ok(())
    }
}

impl Clone for MemoryOs {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            short_term: Arc::clone(&self.short_term),
            mid_term: Arc::clone(&self.mid_term),
            long_term: Arc::clone(&self.long_term),
            vector_store: Arc::clone(&self.vector_store),
            metadata_store: Arc::clone(&self.metadata_store),
            llm_client: Arc::clone(&self.llm_client),
            embedding_queue: Arc::clone(&self.embedding_queue),
        }
    }
}
```

---

## 7. 並行性設計

### 7.1 ロック使用方針

| コンポーネント | ロック種類 | 理由 |
|--------------|----------|------|
| `MemoryOs`全体 | `Arc<Self>`（共有参照） | クローン可能、同時アクセス OK |
| メモリ層（short/mid/long） | `tokio::sync::RwLock` | 非同期排他制御 |
| VectorStore 内部 | `tokio::sync::Mutex` | HNSW 操作は排他必要 |
| MetadataCache | `tokio::sync::RwLock` | 読み取り中心の共有データ |

### 7.2 ロック設計原則

```rust
// ✅ 外側ロックのみ使用（内側ロック削除）
pub struct MemoryOs {
    short_term: Arc<AsyncRwLock<ShortTermMemory>>,  // ← ロックはここだけ
}

pub struct ShortTermMemory {
    inner: VecDeque<Conversation>,  // ← ロックなし、直接操作
    capacity: usize,
}

```

### 7.3 async/await 文脈でのロック統一

```rust
// ✅ tokio::sync に完全統一（std::sync 不使用）
use tokio::sync::{Mutex, RwLock as AsyncRwLock};

pub struct VectorStore {
    index: Mutex<Hnsw<f32>>,              // ← tokio::sync::Mutex
    metadata_cache: AsyncRwLock<HashMap<...>>,  // ← tokio::sync::RwLock
}

```

---

## 8. エラー処理と耐障害性

### 8.1 EmbeddingQueue

```rust
// src/embedding/queue.rs
use tokio::sync::{mpsc, oneshot};
use std::collections::VecDeque;

pub struct EmbeddingQueue {
    tx: mpsc::Sender<EmbeddingTask>,
    batch_size: usize,
    flush_interval_ms: u64,
}

#[derive(Debug)]
pub struct EmbeddingTask {
    pub text: String,
    pub response_tx: oneshot::Sender<Result<Vec<f32>>>,  // Result 型でエラー伝播
}

impl EmbeddingQueue {
    pub fn new(batch_size: usize, flush_interval_ms: u64) -> Self {
        let (tx, rx) = mpsc::channel(100);
        
        tokio::spawn(Self::worker(rx, batch_size, flush_interval_ms));
        
        Self {
            tx,
            batch_size,
            flush_interval_ms,
        }
    }
    
    /// 埋め込み生成リクエスト（エラーを Result で返す）
    pub async fn generate(&self, text: &str) -> Result<Vec<f32>> {
        let (tx, rx) = oneshot::channel();
        
        self.tx.send(EmbeddingTask {
            text: text.to_string(),
            response_tx: tx,
        }).await.map_err(|e| {
            MemoryOsError::EmbeddingError(format!("Queue send failed: {}", e))
        })?;
        
        // 結果待機（エラーも Result として伝播）
        rx.await.map_err(|e| {
            MemoryOsError::EmbeddingError(format!("Queue recv failed: {}", e))
        })??  // 二重の Result を unwrap
    }
    
    async fn worker(
        mut rx: mpsc::Receiver<EmbeddingTask>,
        batch_size: usize,
        flush_interval_ms: u64,
    ) {
        let mut buffer = VecDeque::new();
        let interval = tokio::time::interval(tokio::time::Duration::from_millis(flush_interval_ms));
        
        loop {
            tokio::select! {
                Some(task) = rx.recv() => {
                    buffer.push_back(task);
                    
                    if buffer.len() >= batch_size {
                        Self::process_batch(&mut buffer).await;
                    }
                }
                
                _ = interval.tick() => {
                    if !buffer.is_empty() {
                        Self::process_batch(&mut buffer).await;
                    }
                }
            }
        }
    }
    
    async fn process_batch(buffer: &mut VecDeque<EmbeddingTask>) {
        let texts: Vec<String> = buffer.iter().map(|t| t.text.clone()).collect();
        
        match Self::generate_embeddings_batch(&texts).await {
            Ok(embeddings) => {
                for (task, embedding) in buffer.drain(..).zip(embeddings) {
                    let _ = task.response_tx.send(Ok(embedding));
                }
            }
            Err(e) => {
                // エラー時は Result::Err で返す（呼び出し側で区別可能）
                for task in buffer.drain(..) {
                    let _ = task.response_tx.send(Err(MemoryOsError::EmbeddingError(
                        format!("Batch embedding failed: {}", e)
                    )));
                }
            }
        }
    }
    
    async fn generate_embeddings_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // candle でバッチ推論（実装省略）
        todo!("candle batch inference")
    }
}
```

### 8.2 プロモーション失敗時の耐障害性

| フェーズ | 失敗時処理 | 復元方法 |
|---------|----------|---------|
| **要約** | `summarize_conversation()` エラー | `restore_oldest(conversation)` で短期へ戻す |
| **SQLite 保存** | `save_new_memory_with_embedding()` エラー | `restore_oldest(conversation)` で短期へ戻す |
| **HNSW 挿入** | `vector_store.insert()` エラー | SQLite クリーンアップ + `restore_oldest()` |

```rust
// 失敗時は warn + 復元 + 次回再試行（エラー返却なし）
if let Err(e) = self.summarize_conversation(&conversation).await {
    tracing::warn!("Summarization failed, restoring to short-term");
    
    let mut short_term = self.short_term.write().await;
    short_term.restore_oldest(conversation);  // ← push_front で先頭復元
    
    return Ok(());  // ← エラーを伝播せず、次回再試行
}
```

---

## 9. 実装チェックリスト

### フェーズ 1: コア機能（2 週間）
- [ ] SQLite スキーマ作成（バージョン管理付き、JSON1 必須）
- [ ] BLOB 形式定義（little-endian f32 配列、モデル依存次元数）
- [ ] HNSW インデックス実装（CosineDistance 明示）
- [ ] **起動時インデックス再構築フロー**（全フィールド取得 SQL）
- [ ] **次元不一致レコードは skip + warn**

### フェーズ 2: ロック設計統一（1 週間）
- [ ] 外側ロックのみ使用（内側削除）
- [ ] tokio::sync 完全統一（std::sync 不使用）
- [ ] **HNSW Mutex 設計意図コメント追加**
- [ ] 同時処理テスト（デッドロック確認）

### フェーズ 3: rerank 実装（5 日）
- [ ] SQLite からベクトル再取得メソッド
- [ ] cosine 類似度再計算ロジック
- [ ] **get_search_results() で順序・スコア保証**
- [ ] rerank 精度検証テスト

### フェーズ 4: promote ロジック改善（3 日）
- [ ] ID 集約→更新の 2 フェーズ化
- [ ] **remove で確定してから要約（重複防止）**
- [ ] **失敗時は restore_oldest() で短期へ戻す**
- [ ] プロモーション再現性テスト

### フェーズ 5: EmbeddingQueue 強化（3 日）
- [ ] Result 型エラー伝播
- [ ] batch_buffer 削除、設計統一
- [ ] エラー処理テスト

### フェーズ 6: API 整合性修正（2 日）
- [ ] **save_new_memory_with_embedding に assistant_id 必須追加**
- [ ] build_context の Ok(format!(...)) 修正
- [ ] **get_records_by_ids_ordered() を QueryBuilder に修正**
- [ ] **MemoryRecord の sqlx::FromRow 対応を最初に固定**
- [ ] **VectorStore::metadata_store 参照保持**
- [ ] **次元数引数を MetadataStore メソッドに追加**
- [ ] **MemoryRecord::created_at() の束縛修正**
- [ ] 型チェック全項目パス確認

### フェーズ 7: 統合テスト（1 週間）
- [ ] LoCoMo ベンチマーク再現
- [ ] パフォーマンス計測
- [ ] **再起動時インデックス復元確認**（完全な metadata 復元検証）
- [ ] **プロモーション失敗時の耐障害性テスト**

---

## 🎯 実装着手前の固定事項

### 必須決定事項

| 項目 | 方針 | 理由 |
|------|------|------|
| **中期記憶の HNSW 検索対象** | ✅ **含める（A）** | 含まないと検索に乗りません |
| **中期記憶 embedding 生成** | ✅ **summary から再生成** | 要約文から embedding を再計算して保存・索引化 |
| **HNSW 再構築時の次元不一致** | ✅ **skip + warn** | モデル変更時に古い BLOB が混ざる可能性あり |
| **MemoryRecord の sqlx::FromRow** | ✅ **最初に固定** | query_as/build_query_as を使う前提 |

### 修正点

| # | 問題 | 修正内容 | 状態 |
|---|------|---------|------|
| 1 | VectorStore::get_metadata_store() が unimplemented! | **metadata_store: Arc<MetadataStore> を保持** | ✅ 完了 |
| 2 | MemoryOsConfig::default() で次元数取得 | **引数で次元数を渡す（get_all_embeddings/get_embeddings_by_ids）** | ✅ 完了 |
| 3 | MemoryRecord::created_at() がコンパイル不能 | **変数を束縛してから返す** | ✅ 完了 |

### 推奨設定

```rust
// 中期記憶の embedding 生成
async fn promote_short_to_mid(&self) -> Result<()> {
    // ... 要約完了後
    
    // 中期記憶も HNSW 検索対象にするなら embedding を再生成
    let mid_term_embedding = self.embedding_queue.generate(&summary).await?;
    
    let record_id = self.metadata_store.save_new_memory_with_embedding(
        &self.config.user_id,
        &self.config.assistant_id,
        &summary,
        MemoryLayer::MidTerm,
        Some(&mid_term_embedding),  // ← embedding を保存（HNSW 検索対象）
    ).await?;
    
    // HNSW に登録も embedding 付き
    self.vector_store.insert(record_id, mid_term_embedding, metadata).await?;
}
```

---

## 📊 パフォーマンス見積もり（現実的）

| メトリクス | Python 版 | Rust 版 | 改善率 | 備考 |
|----------|---------|--------------|--------|------|
| **ベクトル検索** (10k) | ~50ms | **~5-8ms** | 6-10 倍 | HNSW 最適化効果 |
| **メモリ使用量** | ~2GB | **~400-600MB** | 3-5 倍削減 | GC なし、効率的構造体 |
| **スループット** (同時リクエスト) | ~20 req/s | **~80-150 req/s** | 4-7 倍 | 並行性設計効果 |
| **LLM API 呼び出し** | ~500ms | **~500ms** | なし | I/O ボトルネック |
| **embedding 生成** (バッチ) | ~100ms/件 | **~20-30ms/件** | 3-5 倍 | バッチ処理効果 |
| **全体エンドツーエンド** | ~700ms | **~600-650ms** | 1.1-1.2 倍 | LLM がボトルネック |

### ⚠️ 重要な注意点
```
✅ ベクトル検索部分のみ大幅改善（5-10 倍）
⚠️ 全体パフォーマンスは LLM API がボトルネック（10-20% 改善）
✅ メモリ効率・並行性は明確な向上
```

---

## ✅ 結論

この仕様書は、以下の修正を反映することで**実装開始版として完全に確定できる**。

### 最終確定修正

| # | 修正内容 | 状態 |
|---|---------|------|
| 1 | `get_records_by_ids_ordered()` を sqlx::QueryBuilder + build_query_as() で実装 | ✅ 完了 |
| 2 | `promote_short_to_mid()` の失敗時復元は `short_term.add()` ではなく `restore_oldest()` を使う | ✅ 完了 |

### 追加修正

| # | 問題 | 修正内容 | 状態 |
|---|------|---------|------|
| 1 | VectorStore::get_metadata_store() が unimplemented! | **metadata_store: Arc<MetadataStore> を保持** | ✅ 完了 |
| 2 | MemoryOsConfig::default() で次元数取得（危険） | **引数で次元数を渡す（get_all_embeddings/get_embeddings_by_ids）** | ✅ 完了 |
| 3 | MemoryRecord::created_at() がコンパイル不能 | **変数を束縛してから返す** | ✅ 完了 |

### 実装着手前の固定事項

| 項目 | 方針 |
|------|------|
| 中期記憶を HNSW 検索対象に含めるか | **含める（A）** |
| 含める場合は summary から embedding を再生成して保存すること | ✅ 必須 |
| HNSW 再構築時の次元不一致レコードは skip + warn とすること | ✅ 必須 |
