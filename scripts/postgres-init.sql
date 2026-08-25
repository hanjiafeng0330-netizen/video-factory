-- 本地栈初始化。容器首次启动时由 postgres 镜像自动执行。
--
-- dev 与 test 使用**不同的库**而不是同一个库的不同 schema：测试会反复清空数据，
-- 共用一个库迟早会把开发中的数据洗掉，而那种丢失是不可恢复的。
-- 库名与 app/platform/config.py 的环境隔离规则一致（vf_{env}）。

CREATE DATABASE vf_test OWNER video_factory;

-- 相似热点、相似脚本、模板与产品匹配用向量检索（设计文档 11 章）。
-- 一期用 pgvector 而不是独立向量库（开发计划 1.1 节）。
\connect vf_dev
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\connect vf_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
