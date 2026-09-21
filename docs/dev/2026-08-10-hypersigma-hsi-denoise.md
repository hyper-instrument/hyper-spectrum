# DG-12：HyperSIGMA HSI 去噪接入与真数据 smoke

## 范围与非目标

本次工作把 WHU-Sigma/HyperSIGMA 的高光谱图像去噪轨接入 HyperSpectrum
统一去噪合同。仓内交付包括显式 HSI 轴语义、固定版本工具清单、外部源码与
权重验证、薄适配器、MAT smoke 入口和 synthetic fixture 测试。真实源码、
权重和 research-use-only 数据只从 Git 工作树外挂载。

本单不建设 ACE Board，不产出正式榜单成绩，不执行训练或微调，不接入分类、
异常检测等其他 HyperSIGMA 任务，不执行 5090 生产 gate，也不访问 hub 数据面。
以下指标只证明固定权重能够在固定真实 patch 上走通统一评测器，不构成模型排名。

## 架构和 HSI 轴合同

- HSI 输入必须是 rank-3、real、dense cube；不接受隐式 channel 维。
- 调用方必须显式给出三个互异轴名，且必须恰好包含 `band` 或 `wavelength`
  之一以及 `y`、`x`；适配器不按 shape 猜轴。
- 官方模型固定接收 `(spectral,y,x)`，其中本数据使用
  `band,y,x` / `index,pixel,pixel`。适配器只做可逆轴置换，不 flatten、
  不切块、不补带、不随机回退。
- 官方去噪模型固定接收 `191x64x64`，输入和输出 shape 必须完全一致。
- 不支持的 rank、complex 或 sparse 输入在运行模型前返回结构化
  incompatibility；评测继续沿用统一 normalized/native RMSE/MAE 合同。

## 固定上游源码身份与模型构造依据

上游源码固定为 `WHU-Sigma/HyperSIGMA` commit
`07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9`，许可证为 Apache-2.0。
运行前逐文件校验以下源码：

```text
ImageDenoising/models/hypersigma/model.py            66c2165b1e04aa7df7c995f3d9ee84a332aa189572364ffb74be8a3b89711c1f
ImageDenoising/models/hypersigma/Spatial.py          b9834e915333c9f7b8c48d818a0a7bf995c6c8966a1f9b1b3299e642bdb2256b
ImageDenoising/models/hypersigma/Spectral.py         c2fbbdcfbf75622c7ecfd88a3ff7c42295f19684e24b6022fd65e1f04050da92
ImageDenoising/models/hypersigma/Spatial_route.py    6b80337dfa94253504936584a85d62b5c81db380ba45ef52bc1a59cf0026883f
ImageDenoising/models/hypersigma/Spectral_route.py   617916efbe01fff98248f7b95bcd53bc9265b14f425aed2f90d0a8ee800931b3
```

模型构造参数直接恢复自固定提交的
`ImageDenoising/models/hypersigma/model.py`；类实现来自同目录的
`Spatial.py`、`Spectral.py`、`Spatial_route.py` 和 `Spectral_route.py`。
适配器在独立临时 namespace 中读取源码后，对即将编译的同一份 bytes 再次
核对固定 SHA-256，不读取或写入额外 `pyc`，构造完成后恢复模块表。官方构造
函数中仅用于预训练初始化的两个
`init_weights` 在锁内临时替换，避免访问未纳入本单的私有 backbone，随后无论
成功或异常都恢复原方法。最终 checkpoint 通过 `strict=True` 载入完整模型，
再冻结参数并进入 eval 模式。

本仓没有复制或 vendor 任一上游源文件，因此不修改
`THIRD_PARTY_NOTICES.md`。

## 双权重身份与同描述符安全加载

两个权重均来自 `WHU-Sigma/HyperSIGMA` revision
`e0567395fbdfddbae994695baf5fc73358a1ec3c`：

| variant | 路径 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| gaussian | `Denoising_models/hypersigma_gaussian_noise_model.pth` | 2266408098 | `dc101cfe7d462d721eb46395b1621d82103cf4e72166d8591b5619cb2d10e806` |
| complex | `Denoising_models/hypersigma_complex_noise_model.pth` | 2266408098 | `8b1162aae6811af67d287b9271e74154db5448c5e2fd6df953dae5d7807bbe7e` |

下载后先由适配器验证固定源码和两个权重，再用 `stat`、`shasum -a 256`
独立复核；两条 smoke 完成后再次计算摘要，前后完全一致。哈希与反序列化使用
同一个已打开文件描述符，避免校验后换文件。加载始终使用
`map_location="cpu"` 和 `weights_only=True`，只在一次加载的局部安全上下文中
允许固定 checkpoint 实际需要的 legacy NumPy scalar、`numpy.dtype` 和
Float64 dtype 类型；没有降级到可执行任意 pickle 的 `weights_only=False`。

复核命令如下，其中两个根目录均指向 Git 工作树外：

```bash
python -m hyperspectrum.adapters.hypersigma --verify \
  --source-root "$HYPERSIGMA_SOURCE_ROOT" \
  --weight gaussian="$DG12_ASSET_ROOT/weights/Denoising_models/hypersigma_gaussian_noise_model.pth" \
  --weight complex="$DG12_ASSET_ROOT/weights/Denoising_models/hypersigma_complex_noise_model.pth"
stat -f '%z %N' "$DG12_ASSET_ROOT"/weights/Denoising_models/*.pth
shasum -a 256 "$DG12_ASSET_ROOT"/weights/Denoising_models/*.pth
```

## research-use-only 数据处理与完整文件 manifest

评测数据固定为 `WHU-Sigma/HyperSIGMA_Datasets` revision
`18ac00c7a98cae281bbdba7f2ed415c1765037d2`。委派方已裁决仅限科研评测
使用；本地数据不入 Git、不公开镜像、不再分发。

固定 revision 的 `Testing/` 下载命令退出 0，共核验 55 个文件：6 个完整
`Cases/test.mat` 和 49 个 `Patch_Cases/test_*.mat`。以下为本地逐字节
manifest，列依次为相对路径、字节数和 SHA-256：

```text
Cases/Case1/test.mat	77022445	d0275b16e30ca373ca628510c9bc15e3fd0debc015379940cf3fd7eeca0591f9
Cases/Case2/test.mat	84492552	643a981365fedfe0ad4b2bb4ddc4530795bc3fc47e70592d32bf5f4d5f898d2b
Cases/Case3/test.mat	84492552	9f3e44cd004951460418d9c94c5968d0f527a3cb0d1a025555f011067a1e3142
Cases/Case4/test.mat	84492552	03384c7204604184f80acffe822091c5e1e13d9a6be872a04e546ad0e485108f
Cases/Case5/test.mat	70515133	505cdf1a6953c7a58bcc5f268a2eaed5d3a8c5b2fbcb5a5667c66c0d4ced7395
Cases/Case6/test.mat	61120264	f4cdba66531601e89189124c83c45312e5d1f9efaebb39a2d8b63d63d72bef28
Patch_Cases/Case1/test_0.mat	9388296	83bf2e16bf4785b80f95b243052697cb54023d9b8afee9ccb6eef3aaacdcc746
Patch_Cases/Case1/test_1.mat	9388296	8a9be82f7e5178b91788c3d34e2ea76f5edce1330cf771186154fb6b8d911aff
Patch_Cases/Case1/test_2.mat	9388296	4ab9ec01d1b39a62f0fb3ff6b547cc9c66a84598d7fcf4064084a2eedb830369
Patch_Cases/Case1/test_3.mat	9388296	db9b235172d326d9d77de592289cc627879ff26f5e4ce512f92d73ce6c793f0f
Patch_Cases/Case1/test_4.mat	9388296	39e9a966a9e0fb572621cbfe7e4f78feade9e2646986a302cf7905e3035697f8
Patch_Cases/Case1/test_5.mat	9388296	8f3eb68ecaa706b4ff0e4e958ccf560078521575cc88ebc224442bd1e76c70cb
Patch_Cases/Case1/test_6.mat	9388296	f0793f5474d8b0b2046d9e48d1466c8699ff5899ed2ed1c4de7c477ebf781f10
Patch_Cases/Case1/test_7.mat	9388296	0a9b12e5b463e8bbcb158194e9585c81344979bf09762597576a8867c706d727
Patch_Cases/Case1/test_8.mat	9388296	d227c19e2d6f278c788bf357d44b8ab4739a5f1f37bbfefdb91afa9da2665f1d
Patch_Cases/Case2/test_0.mat	9388296	aa3a164075b5234d8109acc797626ff25146ecc99e5b77f946473123b31551aa
Patch_Cases/Case2/test_1.mat	9388296	42b069d491d35b57e3873b10666467cbdb27e470606288c31a5b4748065c8d36
Patch_Cases/Case2/test_2.mat	9388296	8561224fb0f3c8b5a6a5f6d0e369cd5a247e2fbefb676e58f0a1eb41aa063c82
Patch_Cases/Case2/test_3.mat	9388296	60669b33dc2719eafa63205eb0b34771daaa7ed6d2f411014f6869eab3284cbc
Patch_Cases/Case2/test_4.mat	9388296	21360998f56eb316a325b91a1c6a6c81af648813ddc9d36ea10b690abf88b8c1
Patch_Cases/Case2/test_5.mat	9388296	79bbcf86d6637d70e79462d6156854272b0ae0a9b066e8c16bab9f5e2297120d
Patch_Cases/Case2/test_6.mat	9388296	0fb99130712908bb381c0fdd097f69785d14e8cfa1e5f3a537d0546177f81eda
Patch_Cases/Case2/test_7.mat	9388296	fb56903dee101088ac28bcbc96c0a63a8d13e9ae65443ec9c56736fffe58f1ac
Patch_Cases/Case2/test_8.mat	9388296	21825b6e5d25e2b4cc6a61850c3f56914bf85a16f2afde0ee26738c8a04acd60
Patch_Cases/Case3/test_0.mat	9388296	07f33ecbcde2811f0820cf912bd986f62394536ac062ec255ca6773128e7877a
Patch_Cases/Case3/test_1.mat	9388296	a185f12efbd197b21c0aa4caa8d36a846ddeb46a3c445455df061c9fd11edc42
Patch_Cases/Case3/test_2.mat	9388296	eab2b54d64df6e785594b5e4d073d88f047a9bfb0c8761f5c0d8e947a54fc87a
Patch_Cases/Case3/test_3.mat	9388296	153437b35b18335f68a85365f6386332983bf81b8f37f984d9ab9904d9179697
Patch_Cases/Case3/test_4.mat	9388296	361cf560744fe480b435afe9f1a38b9244c5518b2fdaf827b8802483f750f3f3
Patch_Cases/Case3/test_5.mat	9388296	abb6b76f111d605e2e0c46dc9400f0931b02736377209446b2bb511ebd4e3354
Patch_Cases/Case3/test_6.mat	9388296	9ca3dcdc9012025624b52810deef79d0ef2e7fd22cc82fdfe6ae19957e3a0aeb
Patch_Cases/Case3/test_7.mat	9388296	1693f523aaa3a53566fad2988dabc93674ec28a3b1cc7225b94d2a6e39b3a2bd
Patch_Cases/Case3/test_8.mat	9388296	72383a5436ec6dffdf7bb6935353e1549dd3ea9642eaf31022774ce808cadb51
Patch_Cases/Case4/test_0.mat	9388296	da7e313dda7384fd14513fc63ce1b16b9f63986f3a95e4ff8c24107196481a03
Patch_Cases/Case4/test_1.mat	9388296	bb956fb46bb056b8ab99c023af09949325406ba88d43651193723e7a0b8f0520
Patch_Cases/Case4/test_2.mat	9388296	0f50069d7aba06272c786811d6e35df955d82632c59d669501d9bace81aed8ac
Patch_Cases/Case4/test_3.mat	9388296	423d56e4e4d96c3c7a5def5cded165217534fa803a340a5af61ad34fc6b2efe7
Patch_Cases/Case4/test_4.mat	9388296	0944551a065ad169201e74ec83a2be8714abc6a620a6cf3655d8aafa7addea54
Patch_Cases/Case4/test_5.mat	9388296	b24003aeb7b32727d04627bdb5ed80f20cda7425e2fe7846c20e4eadd034e22b
Patch_Cases/Case4/test_6.mat	9388296	45f28adcedfa50617e51514659053caddcd44036d19c70257677e872893bd28a
Patch_Cases/Case4/test_7.mat	9388296	c8bd4110073fb77b7f4b14eb62abacad006f1c860ebb5d1be24fb1ebc46c6b61
Patch_Cases/Case4/test_8.mat	9388296	cb975a50da55616de0370b72a6484fdb4915c50bd10e1da3de8152308695947b
Patch_Cases/Case5/test_0.mat	9388296	07d6ec42a886d8e501503890d3921945b1baeb0d68dc530b0d7100b08c75261b
Patch_Cases/Case5/test_1.mat	9388296	85b6ffc1483b59f581921b1c00a7d10de3e2a02ac9aaa340a0cafa936310a695
Patch_Cases/Case5/test_2.mat	9388296	9df7b7e38711dc69e5eb19b80e52ccefacd18d396743e65349d6d7ab04cb1e14
Patch_Cases/Case5/test_3.mat	9388296	d40af8891e3d41dcf57776d7ab34ae3ffaf73db189f606f5e7958d853bfda6d1
Patch_Cases/Case5/test_4.mat	9388296	b3bf8d9257e7f59300e997576793ccb7385be59a61bbf71b30ac950118b08a80
Patch_Cases/Case5/test_5.mat	9388296	3b7b8010da9b43c7200f6e1a53997c11f6b84832e382d755ed55460fe6d9f170
Patch_Cases/Case5/test_6.mat	9388296	7d988adf8e9c0f74acff181516e41d82b0c7693cc21bff99e05eb4606a6ab876
Patch_Cases/Case5/test_7.mat	9388296	175eb3c3c84573ea630783c18c71132274ec149e32f9b878ddfc4396d3790c07
Patch_Cases/Case5/test_8.mat	9388296	4bf86ff685921ce65365340ff6a08b8ae7b9f13a5e5e08c62b4d49a9b810045e
Patch_Cases/Case6/test_0.mat	15280264	1a555e2a01176e9b02f73cd4e1c1ca255e751921057cc4f51db964118e29a9f3
Patch_Cases/Case6/test_1.mat	15280264	1a555e2a01176e9b02f73cd4e1c1ca255e751921057cc4f51db964118e29a9f3
Patch_Cases/Case6/test_2.mat	15280264	3ae51f53ee97c456fd7e8df4b32b09a4b2dfb5e0c30767e8a3152dc54bd1a995
Patch_Cases/Case6/test_3.mat	15280264	a112479cfb423d997197de4bdf96005f37343a3ba22445a262de387fbc4bcdd0
```

完整 `HyperSIGMA_denoising/` 前缀下载还会触及训练库
`Training/wdc.db/data.mdb`。固定 revision 的解析响应声明该对象应为
6,954,654,105 字节，但签发的 CDN 响应实际只有 15,280,264 字节；解析响应
还声明 SHA-256 为
`1a555e2a01176e9b02f73cd4e1c1ca255e751921057cc4f51db964118e29a9f3`，
与 `Patch_Cases/Case6/test_0.mat` 和 `test_1.mat` 的实际摘要相同。默认 Xet、
单线程 Xet 和官方支持的 HTTP 后端均稳定复现同一 consistency error，且本地
没有留下伪装成完整训练库的 `data.mdb`。本单不训练；委派包明列的完整
Testing 评测树已在相同固定 revision 下成功下载并核验。此远端异常不改写为
下载成功，也不通过镜像或再分发绕过。

两条 smoke 输入用 `scipy.io.whosmat` 只读确认：Case1 与 Case5 的
`input`、`gt` 均为实数 `(191,64,64)`；其中 `gt` 为 single、`input` 为
double。没有根据 shape 推断或改写轴。

## 本地环境和依赖版本

真实运行使用 Git 工作树外的独立环境：

```text
macOS-26.5.2-arm64-arm-64bit
Python 3.12.13
torch 2.13.0
timm 1.0.28
einops 0.8.2
CUDA available: false
MPS available: false
device: cpu
```

独立环境的安装没有改写项目 `uv.lock`。运行时出现的是固定上游对 timm 导入
路径、`torch.meshgrid` 和 checkpoint `use_reentrant` 的未来弃用警告；两条
样本没有 failed 或 skipped。

## Case1 / Gaussian smoke

命令：

```bash
/usr/bin/time -l "$DG12_ASSET_ROOT/runtime-venv/bin/python" \
  -m hyperspectrum.adapters.hypersigma_smoke \
  --source-root "$HYPERSIGMA_SOURCE_ROOT" \
  --weight-path "$DG12_ASSET_ROOT/weights/Denoising_models/hypersigma_gaussian_noise_model.pth" \
  --variant gaussian \
  --mat-path "$DG12_ASSET_ROOT/dataset/HyperSIGMA_denoising/Testing/Patch_Cases/Case1/test_0.mat" \
  --input-key input --target-key gt \
  --axis-order band,y,x --axis-units index,pixel,pixel \
  --signal-unit relative_reflectance \
  --sample-id hypersigma-case1-patch0 --group-id hypersigma-case1 \
  --device cpu
```

退出码为 0，状态为 `evaluated`，coverage 为 1.0（1/1，0 failed，0
skipped）。诊断指标：

| 指标 | 值 |
| --- | ---: |
| normalized RMSE | 0.016297917053365495 |
| normalized MAE | 0.012149664642184983 |
| native RMSE (`relative_reflectance`) | 0.05021420549112706 |
| native MAE (`relative_reflectance`) | 0.03743335758755599 |
| real time | 9.83 s |
| maximum resident set size | 2842411008 bytes |
| peak memory footprint | 3434614288 bytes |

## Case5 / complex smoke

这里的 `complex` 是官方复杂噪声权重 variant 名称；输入信号本身仍是合同要求的
real dense cube。

```bash
/usr/bin/time -l "$DG12_ASSET_ROOT/runtime-venv/bin/python" \
  -m hyperspectrum.adapters.hypersigma_smoke \
  --source-root "$HYPERSIGMA_SOURCE_ROOT" \
  --weight-path "$DG12_ASSET_ROOT/weights/Denoising_models/hypersigma_complex_noise_model.pth" \
  --variant complex \
  --mat-path "$DG12_ASSET_ROOT/dataset/HyperSIGMA_denoising/Testing/Patch_Cases/Case5/test_0.mat" \
  --input-key input --target-key gt \
  --axis-order band,y,x --axis-units index,pixel,pixel \
  --signal-unit relative_reflectance \
  --sample-id hypersigma-case5-patch0 --group-id hypersigma-case5 \
  --device cpu
```

退出码为 0，状态为 `evaluated`，coverage 为 1.0（1/1，0 failed，0
skipped）。诊断指标：

| 指标 | 值 |
| --- | ---: |
| normalized RMSE | 0.027158887269897732 |
| normalized MAE | 0.020683199450325346 |
| native RMSE (`relative_reflectance`) | 0.09726106059450106 |
| native MAE (`relative_reflectance`) | 0.07407040999267785 |
| real time | 7.56 s |
| maximum resident set size | 2877440000 bytes |
| peak memory footprint | 3437809120 bytes |

## 自动化验证与相对基线结果

本分支基线为 `origin/main` commit
`ab11693cc3bf1499a4eb3f1e541012a9c7b6b0cb`。开工基线结果：

```text
pytest: 634 passed, 1 skipped, 22 failed, 25 errors
ruff: passed
mypy src: Success（44 个源码文件）
```

基线 pytest 的既有问题有两类：`cu_cha.py` 要求 NumPy 2.4.6，而解析环境为
更新的 NumPy；离线 wheel 测试所需的 `uv-build==0.10.10` 不在本地离线缓存。
最终验证结果：

```text
HyperSIGMA/HSI/registry 聚焦测试：127 passed
含既有离线 wheel 测试的聚焦集合：128 passed, 1 failed, 25 errors
全量 pytest：719 passed, 1 skipped, 22 failed, 25 errors
ruff check .：passed
mypy src：Success（48 个源码文件）
git diff --check：passed
wheel：构建成功、隔离安装成功、安装包静态 verify 成功
wheel package resources：HSI manifest=true，tool schema=true，固定身份=true
Git 资产扩展名泄漏检查：无输出
仓内 50 MiB 以上非 Git 文件检查：无输出
```

全量结果相对基线新增 85 个通过用例，failed、errors、skipped 数量分别保持
22、25、1，故为零新增失败。聚焦集合中的 1 failed 和 25 errors 同样全部来自
既有离线 wheel fixture；单独允许解析固定构建依赖后，实际 wheel 构建、安装、
静态 verify 和 package-resource 检查均退出 0。

## 许可证、不入 Git 与不再分发声明

- HyperSIGMA 代码和权重身份声明为 Apache-2.0；仓库只记录固定身份和薄适配器，
  不复制上游实现。
- 数据集按委派方裁决仅用于 research-use-only 科研评测，不入 Git、不再分发、
  不公开镜像。
- `git ls-files` 对 `.pth`、`.pt`、`.ckpt`、`.mat`、`.h5`、`.hdf5`
  无输出；仓内没有超过 50 MiB 的新增文件。
- 合成单元测试 fixture 明确标记为 `synthetic`，不含真实数据片段。

## 已知限制和未执行事项

- `Training/wdc.db/data.mdb` 存在上述固定 revision 远端对象身份矛盾，本地没有
  该训练库；本单不执行训练，Testing 评测数据完整可用。
- smoke 固定使用两个单 patch，只验证执行链和诊断指标，不代表完整数据集成绩。
- 没有 CUDA 或 MPS 证据；本次按约定仅记录 CPU 结果。
- 未执行 ACE Board、正式榜单、5090 生产 gate、训练、微调或其他任务轨。
- 按委派要求跳过 hub 数据面整章；委派单据回报在 PR 创建后走独立回报接口，
  仓内文档不记录任何凭据或认证值。
