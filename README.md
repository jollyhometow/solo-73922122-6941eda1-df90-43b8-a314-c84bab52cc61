# 月温度距平处理服务（tas Anomaly Service）

将多份模式月温度 NetCDF 加工为**相对同一历史基准的距平**，供后续异常分析。
技术栈：FastAPI + xarray + h5netcdf，重计算运行在**独立子进程**中。

---

## 1. 功能与需求对照

| 需求 | 实现方式 |
| --- | --- |
| 固定历史 NetCDF | 启动时读取 `${DATA_DIR}/baseline/baseline.nc`，启动即校验，配置错误直接启动失败 |
| 上传目标 NetCDF | `POST /api/jobs` 上传，受理前完整校验 |
| 仅含 `tas(time,lat,lon)`、公历月初、开尔文 | `app/netcdf_io.py` 强制校验：唯一数据变量、维度顺序、浮点、`units=K`、Gregorian `datetime64`、每月 1 日、时间不重复、坐标严格单调 |
| 网格完全一致 | 用 lat/lon 坐标值逐格比对（`GridSignature.matches`），不一致返回 400 |
| 按历月逐格忽略缺测求历史均值 | `np.nanmean` 按 1–12 历月分组（`compute_monthly_climatology`） |
| 目标值减对应均值，无基准值则缺测 | 按每个时次的月份索引取对应气候态相减；气候态为 NaN 的格子结果为 NaN |
| 下载保留坐标的 NetCDF | 输出逐字保留目标的 time/lat/lon，h5netcdf 引擎、原子写入 |
| 网页可提交、查状态、取消、下载 | `app/templates/index.html`，1.2 秒轮询 |
| 最多受理两份未结束任务 | `MAX_ACTIVE_JOBS=2`，超出返回 **409**，终态任务释放名额 |
| 并发任务共用未完成的基准计算 | 首个任务创建一个 `BaselineRun`，后续任务挂接同一进程的等待者集合 |
| 取消一份不影响其余 | 仅从等待者集合移除；仍有等待者时共享进程不终止 |
| 最后等待者取消须终止并回收 | 共享进程进入 `draining`，终止实际计算、回收进程、删除临时/成品文件，运行标记 `terminated` |
| 收尾中新任务不得加入旧计算 | 仅 `starting/running` 可挂接；`draining/terminated/error/done` 一律不挂接，新任务另起新运行 |
| 距平在独立进程计算 | `python -I -m app.worker` 子进程，与 Web 进程隔离 |
| 取消与阶段交接/完成交错不启动后续阶段、不发布迟到结果 | 子进程在每个切片与**每个阶段边界**检查取消标志；发布前再检查一次；取消时删除任何 `.part`/成品并以 143 退出 |
| 退出并清理后才报已取消 | 管理器等待子进程退出（reap）并删除任务目录后，才把状态置为 `cancelled` |
| 可实际启动的 Docker Compose | 见下文，含健康检查与持久化卷 |
| 虚构气温与可控慢计算样例 | `scripts/generate_samples.py` 生成数据；`SLOW_STEP_DELAY/SLOW_STEPS` 控制可中断的慢计算 |

---

## 2. 架构与并发模型

```
浏览器 (index.html, 轮询)
        │  HTTP
        ▼
FastAPI (app/main.py)
        │  单一 asyncio.Lock 保护所有状态迁移
        ▼
JobManager (app/manager.py)
   ├── BaselineRun（共享，12 个历月气候态，子进程）
   │      └── waiters={jobA, jobB, ...}
   └── 每个 Job 的 anomaly 子进程（独立）
                │
                ▼
        app/worker.py（python -I，独立进程）
           SIGTERM/SIGINT 或 取消文件 → 置位
           Interrupter 在每个切片/阶段边界检查
```

**状态机**

- 任务：`queued → running → done`；取消路径 `queued/running → cancelling → cancelled`；失败 `→ error`。
- 基准运行：`starting → running → done`；最后等待者取消：`running → draining → terminated`；异常 `→ error`。

**两道取消通道**（保证在受限环境也能演示）

1. `SIGTERM`/`SIGINT`（真实容器的主路径，退出码 143）。
2. 协作式**取消文件** `--cancel-file`：当宿主 seccomp 禁止 `kill(2)`（本开发沙箱即如此），
   管理器捕获 `PermissionError` 后改为 `touch` 取消文件，工作器后台线程每 50ms 轮询，
   置位同一个取消标志，语义与信号完全一致。

**关键正确性**

- 气候态先写到 `climatology.nc.part-<id>`，成功后 `os.replace` 原子发布；未发布前任务持续等待，杜绝迟到结果被消费。
- 子进程在「启动窗口 → 计算 → 发布前」三处设阶段闸门，取消后绝不进入下一阶段。
- 管理器只在**持有锁**的交接点复查取消标志，消除「完成与取消交错」的竞态。

---

## 3. 本地运行（不用 Docker）

需要 Python 3.11+。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # 含 pytest/httpx
pip install -e .

# 1) 生成虚构基准与样例目标
python -I -m scripts.generate_samples --data-dir ./data

# 2) 启动（慢计算便于观察取消）
DATA_DIR=./data SLOW_STEP_DELAY=0.35 SLOW_STEPS=12 \
  python -I -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000 。

---

## 4. Docker Compose 运行（推荐演示）

```bash
docker compose up --build
```

- 首次启动 entrypoint 自动在卷 `tas-data` 中生成虚构基准与两个样例目标。
- 服务：http://localhost:8000 ，健康检查 `GET /healthz`。
- 慢计算参数已在 compose 中配好（`SLOW_STEP_DELAY=0.35`、`SLOW_STEPS=12`）。

重置演示数据：

```bash
docker compose down -v        # 删除卷，下次启动重新生成
```

---

## 5. 浏览器演示脚本（共享计算 + 交错取消）

1. 打开首页，确认「历史基准状态」显示网格、气候态尚未生成。
2. 在「④ 样例数据」下载 `target_sample_a.nc`、`target_sample_b.nc`。
3. 连续上传 **a、b 两份**（动作要快，慢计算约十几秒）。
   - 「当前运行」出现一个 `running` 的共享基准，等待者同时包含两个任务。
4. 再尝试上传第三份 → 直接 **409**（前端弹错，最多两份未结束任务）。
5. 取消其中 **a**：
   - a 很快变为 `cancelled`（子进程已退出、目录已清理）；
   - 共享基准仍是 `running`，b 不受影响继续等待并最终 `done`。
6. 重新开两份任务，在基准完成前**依次取消，最后取消最后一个等待者**：
   - 共享运行变为 `draining → terminated`，气候态文件与 `.part` 被删除；
   - 此后再提交新任务，会生成一个**全新 run id** 的运行，绝不加入旧计算。
7. 对 `done` 任务点「下载距平 NetCDF」，文件名 `<原名>_anomaly.nc`。

命令行等价演示：

```bash
curl -X POST -F "file=@data/samples/target_sample_a.nc;type=application/x-netcdf" \
  http://localhost:8000/api/jobs
curl -X POST http://localhost:8000/api/jobs/<id>/cancel
curl -OJ http://localhost:8000/api/jobs/<id>/download
```

---

## 6. HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 健康检查 |
| GET | `/` | 演示网页 |
| GET | `/api/baseline` | 固定基准信息、网格、当前共享运行与等待者 |
| GET | `/api/samples` | 可下载样例列表 |
| GET | `/samples/{name}` | 下载样例 NetCDF |
| POST | `/api/jobs` | multipart 上传字段 `file`，201 受理；400 校验失败；409 超并发 |
| GET | `/api/jobs` | 全部任务与活跃计数 |
| GET | `/api/jobs/{id}` | 单任务状态 |
| POST | `/api/jobs/{id}/cancel` | 请求取消（终态任务幂等） |
| GET | `/api/jobs/{id}/download` | 下载距平 NetCDF（非 done 返回 409） |

任务状态：`queued / running / cancelling / done / error / cancelled`。

---

## 7. 配置项（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `/data` | 数据根目录（基准、任务、样例） |
| `MAX_ACTIVE_JOBS` | `2` | 未结束任务上限 |
| `SLOW_STEP_DELAY` | `0.25` | 每个慢切片秒数 |
| `SLOW_STEPS` | `12` | 慢切片数量（乘积为总人造延迟） |
| `WORKER_TERM_TIMEOUT` | `8` | 优雅退出等待秒数，超时升级 SIGKILL |
| `POLL_INTERVAL` | `0.15` | 内部轮询间隔 |

---

## 8. 测试

```bash
pip install -r requirements-dev.txt
pytest
```

39 个测试覆盖：

- 数据契约校验（重复时间、非月初、非公历日历、非开尔文、网格不一致、维度顺序、额外变量、非单调坐标、整型 tas、损坏文件）；
- 气候态忽略缺测、距平数值、无基准月→NaN、坐标逐字保留；
- 子进程 CLI 正常路径、SIGTERM/取消文件两条取消通道、启动窗口取消、无残留文件；
- 管理器：端到端、两份共享一个基准进程、409 上限、取消单个不影响其余、
  最后等待者取消终止回收且新任务不加入旧运行、距平阶段取消回收、
  完成与取消交错不发布迟到结果、基准失败报错、关闭时回收全部进程；
- HTTP 层完整上传/状态/取消/下载流程。

> 注：本开发沙箱的 seccomp 禁止 `kill(2)`，信号相关用例自动改走取消文件通道；
> 在普通 Docker 容器内则走真实 SIGTERM，两条路径均在 CI 风格的测试中验证。

---

## 9. 目录结构

```
app/
  config.py            环境变量配置
  models.py            状态枚举与退出码
  netcdf_io.py         NetCDF 校验与网格签名
  compute.py           气候态/距平计算、可中断慢计算、虚构样例生成
  worker.py            独立进程工作器 CLI（信号 + 取消文件）
  manager.py           任务/基准生命周期、并发上限、共享与取消语义
  main.py              FastAPI 路由
  templates/index.html 演示网页
scripts/
  generate_samples.py  生成虚构基准与样例
docker/
  entrypoint.sh        容器入口（按需生成样例后启动服务）
tests/                 pytest 测试
Dockerfile
docker-compose.yml
```
