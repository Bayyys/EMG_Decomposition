# EMG 分解

![](https://img.shields.io/pypi/v/emgdecomp) [![DOI](https://zenodo.org/badge/423892522.svg)](https://zenodo.org/徽章/latestdoi/423892522）

用于将 EMG 信号分解为运动单元触发的软件包，为 Formento 等人 2021 年创建。很大程度上基于 Negro 等人，2016 年。通过 CUDA 支持 GPU，通过 Dask 支持分布式计算。

## 安装

```shell
pip install emgdecomp
```

对于那些想要使用 [Dask](https://dask.org/) 和/或 [CUDA](https://cupy.dev/) 的人，您也可以运行：

```shell

pip install emgdecomp [dask]

pip install emgdecomp[cuda]

```

## 用法

### 基本的

```python
# 数据应该是一个 n_channels x n_samples 的 numpy 数组
sampling_rate, data = fetch_data(...)

decomp = EmgDecomposition(
  params=EmgDecompositionParams(
    sampling_rate=sampling_rate
  ))

firings = decomp.decompose(data)
print(firings)
```

生成的 `firings` 对象是一个 NumPy 结构化数组，包含 `source_idx`、`discharge_samples` 和 `discharge_seconds` 列。 `source_idx` 是从数据中学习到的每个“源”的 0 索引 ID；每个来源都是推定的运动单位。 `discharge_samples` 表示源被检测为“正在发射”的样本；请注意，该算法只能检测到延迟的源。 `discharge_seconds` 是通过传入的采样率将 `discharge_samples` 转换为秒。

作为一个结构化的 NumPy 数组，生成的 `firings` 对象适合转换为 Pandas DataFrame：

```python
import pandas as pd
print(pd.DataFrame(firings))
```

并且可以通过 `decomp.model` 属性根据需要询问“来源”（即对应于运动单元的组件）：

```python
model = decomp.model
print(model.components)
```

### 先进的

给定一个已经合适的 `EmgDecomposition` 对象，然后您可以通过 `transform` 分解一批新的 EMG 数据及其现有源：

```python

# 假设 decomp 已经适合
new_data = fetch_more_data(...)
new_firings = decomp.transform(new_data)
print(new_firings)
```

或者，您可以添加新来源（即新的假定运动单位），同时使用 `decompose_batch` 保留现有来源：

```python
# 假设 decomp 已经适合
more_data = fetch_even_more_data(...)
# 对应于现有和新添加的源的触发
firings2 = decomp.decompose_batch(more_data)
# 应该有至少和之前一样多的组件 decompose_batch()decompose_batch()
print(decomp.model.components)
```

最后，还包括基本的绘图功能：

```python
from emgdecomp.plots import plot_firings, plot_muaps
plot_muaps(decomp, data, firings)
plot_firings(decomp, data, firings)
```

### 文件 I/O

`EmgDecomposition` 类配备了 `load` 和 `save` 方法，可以根据需要将参数保存/加载到磁盘；例如：

```python
with open('/path/to/decomp.pkl', 'wb') as f:
  decomp.save(f)

with open('/path/to/decomp.pkl', 'rb') as f:
  decomp_reloaded = EmgDecomposition.load(f)
```

### Dask 和/或 CUDA

EmgDecomposition 支持 Dask 和 CUDA，以支持跨工作人员的分布式计算和/或使用 GPU 加速。每个都通过 `EmgDecomposition` 构造函数中的 `use_dask` 和 `use_cuda` 布尔标志控制。

### 参数调优

请参阅 [EmgDecompositionParameters](https://github.com/carmenalab/emgdecomp/blob/master/emgdecomp/parameters.py) 中的参数列表。 `master` 的默认值设置为用于 Formento 等。 al, 2021 并且对于其他人来说应该是合理的默认值。

具体如下：

保存与EMG元数据相关的参数。分开在一个单独的类中，以便于持久化/加载过去的运行。

属性： 

sampling_rate (float): 数据的采样率，以赫兹为单位。

 Extension_factor（int）。给每个数据点增加多少个时间滞后。Negro 2016建议扩展参数等于1000/m，其中m是记录中的通道数量。注意，这个参数是元数据准确性和计算负荷之间的权衡。 

maximum_num_sources（int）：要找到的最大数量的源。由于这个元数据只能找到延迟小于扩展因子的源，所以找到的源总数可能高于通道数。 

min_peaks_distance_ms（Optional[float]）。在改进迭代过程中，在查找峰值算法中使用的最小距离--以毫秒计。如果没有，则使用所有的峰值（默认=10ms）。 

contrast_function（str）：和sklearn.metadata.FastICA中一样，这是G'函数的函数形式，用于近似负熵。可以是 "logcosh"，"exp"，或 "cube "或 "square"。请注意，'square'是相当少用的，但在这里被包括在内，因为在Negro 2016中使用了。

max_iter（int）：计算每个源的ICA的最大迭代次数。 

Convergence_tolerance（float）。用于确定定点ICA迭代和针对ISI优化的改进迭代的收敛的容忍度。 

sil_threshold（float）：Silhoutte Score（SIL）的阈值，超过这个阈值的源被认为是 "好的"，并添加到我们的结果中。Negro 2016使用的阈值是0.9。将此设置为1.0或更大，以跳过SIL的计算。 

regularization_method（str）：美白过程中的正则化方法。可以是。         - 截断"，在噪声阈值下的evals被截断为零 - "添加"，在白化过程中，噪声阈值被添加到每个特征值上 

regularization_factor_eigvals（浮float）：数据矩阵的特征值的一部分被视为噪声；特征值以升序排列，这些特征值的这部分平均值在白化过程中被用作正则化因子。Negro 2016使用的因子是0.5。 

improvement_iteration_metric（str）："sil"、"isi"、"csm1"、"csm3 "之一。指的是在 "改进迭代"（fastICA之后的迭代）中计算的指标。Negro 2016和其他人使用了尖峰间隔的变异系数（CoV ISI）；他们的实验利用了在特定%力下保持的等长收缩，因此在他们所有的检测单位中期待一致的ISI是合理的。对于更多的变量实验，如自由运动，SIL可能是一个更好的选择，以普遍强调 "信号 "与 "噪音"。

 improvement_iteration_min_peak_heights（Optional[float]）：在改进迭代步骤中用于检测峰值的最小峰值高度。根据Negro2016，这应该是无--意味着所有的峰都被考虑在内。然而，K-means++并不总是能很好地划分出什么是信号与噪音。设置一个最小峰高可能有助于解决这个问题（如果需要的话，一个启发式发现的好的起始值是0.9）。 

fraction_peaks_initialization（float）：使用从白化数据中随机选择的峰进行规范化的源的比例。

 firings_similarity_metric（str）："perc_coincident "或 "spike_distance "之一。spike_distance "使用http://mariomulansky.github.io/PySpike/ 中实现的SPIKE-distance度量。请参考这篇文章http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony，以获得一个快速的概述。这是用来比较两个不同的假定来源的指标，看它们是否相互重复。

max_similarity (float)：介于0和1之间的值，定义了两个尖峰序列需要有多大的相似度才能被认为是来自同一个源。这影响到分解后哪些源被保留。 

min_n_peaks (int): 一个源需要有的最小检测峰数，以便被视为一个运动单元。这用于在分解算法的清洁步骤中过滤实际来源的噪声。

 w_init_indices（Optional[list]）：用于初始化前n个来源的指数列表（n = len(list)）。当所有提供的指数都用完后，算法会回到标准行为：使用峰值，直到'fraction_peaks_initialization'，然后随机初始化数值。 

waveform_duration_ms（float）。分解结束时要提取的波形的持续时间，以保存每个源的平均muap波形。 

pre_spike_waveform_duration_ms（可选[float]）。如果提供，用于确定提取的muap波形相对于检测到的尖峰的偏移。如果没有，则提取的波形以尖峰为中心。 

clustering_algorithm（str）："kmeans "或 "ward "之一（默认为Negro 2016的kmeans）。这定义了用于从投影数据中检测运动单元动作电位的聚类算法。请注意，分解算法使用尖峰检测来细化计算的成分。因此，这个参数对分解性能有重大影响。虽然'kmeans'是Negro 2016年建议的，并且是这里的默认算法，但这个算法假设是均匀的集群（集群需要有类似的方差）。在控制较少/噪音较大的情况下，这一假设基本上没有得到满足。沃德 "是一种聚类技术，允许不均匀的聚类大小（见https://scikit-learn.org/stable/modules/clustering.html)因此更适用于异常值（即尖峰）的检测。请注意，在线分类的阈值总是被设置为噪声和尖峰集群之间的平均值（即在线分类执行 "kmeans"，其平均值使用这里选择的算法的聚类结果计算）。另外，值得考虑的是，kmeans大约比ward快2倍。 

dask_chunk_size_samples（int）：Dask中每个chunk的样本数。为了在Dask中获得最佳性能，最终的块数应该在~100MB-3GB之间。

sil_max_samples (int): 计算SIL时的随机子集的大小（完整的SIL需要O(n^2)内存/计算）。留出负数以对整个数据集进行采样。

## 文档

有关更多详细信息，请参阅类 `EmgDecomposition` 和 `EmgDecompositionParameters` 的文档。

## 致谢

如果您喜欢此软件包并将其用于您的研究，您可以：

- 引用神经工程杂志论文，Formento 等。 al 2021，为此开发了此软件包：10.1088/1741-2552/ac35ac

- 使用其 DOI 引用此 github 存储库：10.5281/zenodo.5641426

- 使用右上角的星号按钮为这个 repo 加注星标。

## 贡献/问题

如果有问题或功能请求，请随时在此项目中打开问题。非常鼓励对功能请求的拉取请求，但请在实施之前随意创建一个问题，以确保所需的更改听起来合适。



# EMGDecomp

![](https://img.shields.io/pypi/v/emgdecomp) [![DOI](https://zenodo.org/badge/423892522.svg)](https://zenodo.org/badge/latestdoi/423892522)

Package for decomposing EMG signals into motor unit firings, created for Formento et al 2021. Based heavily on Negro et al, 2016. Supports GPU via CUDA and distributed computation via Dask.


## Installation

```bash
pip install emgdecomp
```

For those that want to either use [Dask](https://dask.org/) and/or [CUDA](https://cupy.dev/), you can alternatively run:

```bash
pip install emgdecomp[dask]
pip install emgdecomp[cuda]
```

## Usage

### Basic

```python
# data should be a numpy array of n_channels x n_samples
sampling_rate, data = fetch_data(...)

decomp = EmgDecomposition(
  params=EmgDecompositionParams(
    sampling_rate=sampling_rate
  ))

firings = decomp.decompose(data)
print(firings)
```

The resulting `firings` object is a NumPy structured array containing the columns `source_idx`, `discharge_samples`, and `discharge_seconds`. `source_idx` is a 0-indexed ID for each "source" learned from the data; each source is a putative motor unit. `discharge_samples` indicates the sample at which the source was detected as "firing"; note that the algorithm can only detect sources up to a delay. `discharge_seconds` is the conversion of `discharge_samples` into seconds via the passed-in sampling rate.

As a structured NumPy array, the resulting `firings` object is suitable for conversion into a Pandas DataFrame:

```python
import pandas as pd
print(pd.DataFrame(firings))
```

And the "sources" (i.e. components corresponding to motor units) can be interrogated as needed via the `decomp.model` property:

```python
model = decomp.model
print(model.components)
```

### Advanced

Given an already-fit `EmgDecomposition` object, you can then decompose a new batch of EMG data with its existing sources via `transform`:

```python
# Assumes decomp is already fit
new_data = fetch_more_data(...)
new_firings = decomp.transform(new_data)
print(new_firings)
```

Alternatively, you can add new sources (i.e. new putative motor units) while retaining the existing sources with `decompose_batch`:

```python
# Assumes decomp is already fit

more_data = fetch_even_more_data(...)
# Firings corresponding to sources that were both existing and newly added
firings2 = decomp.decompose_batch(more_data)
# Should have at least as many components as before decompose_batch()
print(decomp.model.components)
```

Finally, basic plotting capabilities are included as well:

```python
from emgdecomp.plots import plot_firings, plot_muaps
plot_muaps(decomp, data, firings)
plot_firings(decomp, data, firings)
```

### File I/O
The `EmgDecomposition` class is equipped with `load` and `save` methods that can save/load parameters to disk as needed; for example:

```python
with open('/path/to/decomp.pkl', 'wb') as f:
  decomp.save(f)

with open('/path/to/decomp.pkl', 'rb') as f:
  decomp_reloaded = EmgDecomposition.load(f)
```

### Dask and/or CUDA
Both Dask and CUDA are supported within EmgDecomposition for support for distributed computation across workers and/or use of GPU acceleration. Each are controlled via the `use_dask` and `use_cuda` boolean flags in the `EmgDecomposition` constructor.

### Parameter Tuning

See the list of parameters in [EmgDecompositionParameters](https://github.com/carmenalab/emgdecomp/blob/master/emgdecomp/parameters.py). The defaults on `master` are set as they were used for Formento et. al, 2021 and should be reasonable defaults for others.

## Documentation
See documentation on classes `EmgDecomposition` and `EmgDecompositionParameters` for more details.

## Acknowledgements
If you enjoy this package and use it for your research, you can:

- cite the Journal of Neural Engineering paper, Formento et. al 2021, for which this package was developed: 10.1088/1741-2552/ac35ac
- cite this github repo using its DOI: 10.5281/zenodo.5641426
- star this repo using the top-right star button.

## Contributing / Questions

Feel free to open issues in this project if there are questions or feature requests. Pull requests for feature requests are very much encouraged, but feel free to create an issue first before implementation to ensure the desired change sounds appropriate.
