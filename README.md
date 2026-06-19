# Movie Streaming Recommendation Lab

本项目实现算法课程题目 23：电影流媒体用户行为排序与推荐系统。当前版本可以作为算法作业第一版提交：默认使用 MovieLens small 数据集完成可运行的前端演示、后端 API、命令行实验、算法对比和单元测试；Netflix Prize 作为进阶数据集，提供大规模评分导入、排序、标题检索、协同过滤、矩阵分解、候选召回、贝叶斯搜索、Learning-to-Rank 与神经重排实验。

## 当前完成度

- MovieLens 默认演示完整可跑：Top-N 排序、标题/类型/标签搜索、相似电影推荐、个性化 For You 推荐、标签语义邻居和运行时间对比。
- 前端与后端已打通：`start_frontend.ps1` 会创建虚拟环境、安装轻量依赖、启动 FastAPI，并打开本地看板。
- 算法作业核心内容已覆盖：排序、Top-N 堆选择、线性检索、倒排索引、相似度推荐、协同过滤、评分平滑与运行时间实验。
- Netflix Prize 已作为进阶模块隔离在 `src/datasets/netflix/`，不会影响默认 MovieLens 演示。
- 测试覆盖 MovieLens API smoke、双数据集 API、搜索、协同过滤、评分修正、Netflix 矩阵分解、候选召回、贝叶斯搜索、Learning-to-Rank 和神经重排等路径。

## 一键演示

需要 Python 3.10+。第一次运行会自动创建 `.venv`、安装 `requirements.txt`、下载 MovieLens small 数据集，并启动本地页面。

Windows:

```powershell
.\start_frontend.ps1
```

也可以直接双击：

```text
start_demo.bat
```

macOS / Linux:

```bash
chmod +x ./setup_venv.sh ./run.sh ./start_frontend.sh
./start_frontend.sh
```

浏览器地址：

```text
http://127.0.0.1:8013/
```

演示时保持终端窗口打开。按 `Ctrl+C` 或关闭窗口即可停止后端。

## 数据集

默认数据集会自动下载：

```text
data/ml-latest-small
```

启动脚本会在缺少数据时自动从 GroupLens 下载 `ml-latest-small.zip`，并解压出 `movies.csv`、`ratings.csv`、`tags.csv`。这是第一版课堂演示和基础实验的主数据集。

进阶数据集：

```text
data/netflix-prize/download
```

Netflix Prize 原始数据较大，不提交到 Git。放入 `movie_titles.txt` 和 `training_set/mv_*.txt` 后，先构建本地 DuckDB：

```powershell
python -m src.datasets.netflix.import_duckdb --force
python -m src.datasets.netflix.scoring
```

生成文件：

```text
data/netflix-prize/netflix.duckdb
output/netflix/movie_scores.csv
```

说明：Netflix Prize 数据集没有 genre 和 tag 字段，因此 Netflix 只支持标题搜索；genre/tag 搜索会明确返回不支持。

如果用 `.\start_frontend.ps1 -Dataset netflix` 或 `.\run.ps1 --dataset netflix ...`，脚本会检查 `data/netflix-prize/netflix.duckdb`。若数据库不存在但本地原始文件已经放好，会自动构建 DuckDB；若原始文件也没有，会给你 8 秒确认是否下载/解压 Netflix Prize 大包，默认不下载。

Netflix 下载行为：

- 直接按回车、输入 `n`、或 8 秒不操作：跳过下载并停止 Netflix 启动。
- 输入 `y` 再回车：下载或使用本地缓存的 `data/nf_prize_dataset.tar.gz`，解压后构建 `netflix.duckdb`。
- 可用环境变量 `NETFLIX_PRIZE_URL` 覆盖默认下载地址。
- 默认下载源是公开镜像 `https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz`，文件较大，首次处理会比较久。

## 运行命令

Windows:

```powershell
.\run.ps1 demo
.\run.ps1 top -n 10 --algorithm heap
.\run.ps1 top -n 10 --algorithm merge
.\run.ps1 search title "Toy Story"
.\run.ps1 search genre Comedy
.\run.ps1 search tag funny
.\run.ps1 recommend "Toy Story"
.\run.ps1 experiment
```

macOS / Linux:

```bash
./run.sh demo
./run.sh top -n 10 --algorithm heap
./run.sh search title "Toy Story"
./run.sh recommend "Toy Story"
./run.sh experiment
```

指定 Netflix：

```powershell
.\run.ps1 --dataset netflix demo
.\run.ps1 --dataset netflix top -n 10 --algorithm heap
.\run.ps1 --dataset netflix search title "Matrix"
.\run.ps1 --dataset netflix recommend "The Matrix"
```

前端也可以切换默认数据集：

```powershell
.\start_frontend.ps1 -Dataset movielens
.\start_frontend.ps1 -Dataset netflix -Port 8014
```

## API

启动前端脚本后，FastAPI 与静态前端在同一个端口：

```text
http://127.0.0.1:8013/docs
```

主要接口：

- `GET /api/health`：健康检查和可用数据集。
- `GET /api/dashboard?dataset=movielens`：数据集摘要与实验运行时间。
- `GET /api/top?n=10&algorithm=heap`：Top-N 堆选择排序结果。
- `GET /api/top?n=10&algorithm=merge`：归并排序结果。
- `GET /api/top?n=10&score_mode=preference_adjusted`：用户打分宽严修正后的排行榜。
- `GET /api/search?kind=title&query=Toy%20Story`：索引搜索。
- `GET /api/recommend?title=Toy%20Story`：相似电影推荐。
- `POST /api/events`：记录浏览器会话行为。
- `GET /api/for-you?session_id=...`：个性化推荐。
- `GET /api/tag-semantics?tag=funny`：MovieLens 标签语义邻居。

所有主要接口都支持 `dataset=movielens` 或 `dataset=netflix` 查询参数。Netflix 的 tag/genre 相关接口会返回不支持，因为原始数据没有这些字段。

## 实现的算法

排序与 Top-N：

- Merge Sort：完整排序，复杂度 `O(n log n)`。
- Heap Sort：完整堆排序，复杂度 `O(n log n)`。
- Top-N Heap Selection：只保留前 N 个候选，适合推荐榜单。

检索：

- Linear Search：作为基线。
- Dictionary / Inverted Index Search：标题、类型、标签的索引检索。
- Netflix Title Search Adapter：复用 MovieLens 标题索引思想，但不伪造 genre/tag 能力。

推荐：

- 内容相似推荐：综合共享类型、共享标签和综合评分。
- MovieLens For You：浏览器行为、稀疏内容向量、倒排索引余弦召回、历史评分用户协同过滤和冷启动兜底。
- Netflix 协同过滤：基于用户评分相似度的推荐和相似电影推荐。
- Netflix 矩阵分解：带偏置的隐因子模型与时间切分评估。
- Netflix 多路候选召回：popular quality、MF user top、profile centroid、item-item CF、user-user CF、year affinity。
- Netflix Learning-to-Rank / Neural Reranker：对候选集进行二阶段重排实验。

评分：

- Bayesian rating：抑制低样本电影的偶然高分。
- Preference-adjusted rating：修正不同用户打分宽严差异。
- Comprehensive score：结合评分质量、热度、近期活跃度，MovieLens 还结合标签活跃度。

## 实验输出

MovieLens 输出在：

```text
output/movielens/
```

常见文件：

- `movie_profiles.csv`
- `sorting_runtime.csv`
- `search_runtime.csv`
- `runtime_chart.svg`
- `tag_semantics.json`

Netflix 输出在：

```text
output/netflix/
output/netflix_large_search/
output/netflix_bayes_search/
output/netflix_learning_to_rank/
output/netflix_neural_reranker/
```

常见文件：

- `movie_scores.csv`
- `sorting_runtime.csv`
- `matrix_factorization_metrics.csv`
- `recommender_comparison.csv`
- `candidate_recall.csv`
- `best_params.json`

## 进阶实验命令

进阶实验依赖独立放在 `requirements-training.txt` 和 `requirements-tuning.txt`，避免默认演示安装过重。

```powershell
.\setup_venv.ps1 -Requirements requirements-training.txt
.\run.ps1 --dataset netflix large-search --profile smoke
.\run.ps1 --dataset netflix bayes-search --profile smoke --mode hybrid
.\run.ps1 --dataset netflix rerank --profile smoke
.\run.ps1 --dataset netflix neural-rerank --profile smoke
```

GPU PyTorch 是可选项：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
```

## 测试

运行全部测试：

```powershell
python -m unittest discover
```

只跑 MovieLens HTTP smoke：

```powershell
python -m unittest tests.test_movielens_http_smoke
```

只跑双数据集 HTTP smoke：

```powershell
python -m unittest tests.test_multi_dataset_http_smoke
```

说明：部分 HTTP smoke 会在缺少本地数据集或 Netflix DuckDB 时自动跳过，这是为了让项目在没有大数据文件的机器上仍可运行基础测试。

## 项目结构

```text
src/
  algorithms/        # 通用排序算法
  core/              # 数据集注册、路径和 pipeline 抽象
  datasets/
    movielens/       # 默认课堂演示数据集
    netflix/         # 进阶大规模推荐实验
  api.py             # FastAPI 后端
  main.py            # 命令行入口
web/                 # 本地前端看板
tests/               # 单元测试和 HTTP smoke
docs/                # 报告提纲
output/              # 实验生成结果
data/                # 本地数据集，不提交大文件
```

## 第一版报告建议

报告可以按下面顺序展开：

1. 问题背景：电影流媒体平台需要根据评分和行为快速排序、检索、推荐。
2. 数据预处理：MovieLens 的电影、评分、标签如何聚合成电影画像。
3. 排序算法：归并排序、堆排序、Top-N 堆选择的复杂度和实验对比。
4. 检索算法：线性检索与倒排索引检索的速度差异。
5. 推荐算法：内容相似、行为个性化、协同过滤的设计。
6. 系统展示：前端看板、API、命令行实验输出。
7. 扩展工作：Netflix Prize 大规模推荐实验作为加分和后续优化方向。

第一版提交时，优先演示 MovieLens 路径；Netflix 模块适合作为“已完成的进阶扩展”，根据现场时间选择是否展示。
