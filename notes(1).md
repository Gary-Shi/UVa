# 高斯图模型

## High Dimensionality Graphs and Variable Selection with the LASSO

定义：势函数为多元高斯分布的马尔可夫网络，由马尔可夫网络的性质可知，$q_{ij}=0 \Leftrightarrow X_i$与$X_j$之间满足局部马尔可夫性$\Leftrightarrow$ $X_i$和$X_j$不相邻

方法：

- Covariance Selection: Inefficient, consistent only when $n>>p$
- Neighborhood Selection: Lasso Regression, Consistency with High Dimensionality

NS的基本思路：

作$X_i$关于$X_{-i}$的最小二乘回归，得到线性回归系数向量$\theta^i$

另一方面，由分块矩阵求逆得到$p(X_i|X_{-i})=N(-Q_i^TX_{-i}/q_{ii}, Q_{i|-i})$

所以$\theta^i_j=-\frac{q_{ij}}{q_{ii}}$，$\theta^i_j=0\Leftrightarrow q_{ij}=0$

NS with lasso: 作lasso回归

注意点：

通过交叉检验得到的$\lambda_{cv}$不好，没有consistency

Asymptotics: 略

数值实验Graph的生成：

在$[0,1]\times [0,1]$上均匀取$p$个点，每两个点之间有连线的概率为$\varphi(d/\sqrt p)$，$d$是距离，$\varphi$是标准正态分布，每个点最多4条边，否则随机删除多余的。$q_{ii}=1, q_{ij}=0$如果ij不相邻，$q_{ij}=0.245$如果ij相邻（保持对角占优）。然后得到$\Sigma$，注意乘系数让对角元为1，再用$L^T L$分解抽样得到$X=L^TU$



## Hyper Inverse Wishart Distribution for Non-decomposable Graphs and its Application to Bayesian In ference for Gaussian Graphical Models

摘要：提出对不可分图使用DY共轭后验，当G为一个不完全的素图时，这个后验就是逆威沙特分布的一个推广。用重要性采样可以算出归一化系数。对不可分zero-pattern提出不完全三角矩阵(incomplete triangular matrix)的三角补全法(triangular completion)。

一个图G可以连续地分解成它的不可分分量(prime components)，这些分量都是完全的，所以可分解图直接能应用针对G的极大团的饱和边缘模型理论（无向图模型的势函数可以分解成极大团的势函数的乘积，逆威沙特分布就是饱和模型(完全图)的共轭后验）

记$\mathscr{W}$为G对应的完全图，$G = (V, \mathscr{V})$，$\overline{\mathscr{V}} = \mathscr{W}-\mathscr{V} $

$\mathscr{V}$-Incomplete Matrix: 记为$\Gamma$，在(i,j)有边的位置填$\gamma_{ij}$，在没有边的位置填$*$

集合$M_*^+(G)$包含所有**某种方式填满后能成为正定矩阵**的$\mathscr{V}$-Incomplete Matrices

集合$M^+(G)$包含所有满足$\Gamma^{-1}(i,j)=0 \Leftrightarrow (i,j)\in \overline{\mathscr{V}}$的$\Gamma^{-1}$矩阵

已证：对于任意G，存在唯一的矩阵$\Gamma$使得它是$\Gamma^{\mathscr{V}}$的补全矩阵，同时$\Gamma^{-1}\in M^+(G)$

这篇文章的改进：由于$\Gamma$正定对称，所以可以只取上三角部分$\Phi$为$\mathscr{V}$-Incomplete Triangular Matrix

证明了：对于任意G，存在唯一的上三角矩阵$\Phi$使得它是$\Phi^{\mathscr{V}}$的补全矩阵，又有$\Gamma^{-1}=\Phi^T\Phi\in M^+(G)$

TBC.



## Decomposable Gaussian Graphical Models

decomposable高斯图模型的分解：
$$
p(x | \Sigma, g)=\frac{\prod_{C \in \mathcal{C}} p\left(x_{C} | \Sigma_{C}\right)}{\prod_{S \in \mathcal{S}} p\left(x_{S} | \Sigma_{S}\right)}
$$

$$
p\left(x_{C} | \Sigma_{C}\right)=(2 \pi)^{-n|C| / 2}\left(\operatorname{det}\left(\Sigma_{C}\right)\right)^{-n / 2} \exp \left\{-\frac{1}{2} \operatorname{tr}\left(S_{C}\left(\Sigma_{C}\right)^{-1}\right)\right\}
$$

参数估计和模型选择的基础. C是图中的极大团(Clique)，S是图分解式中的Seperator。

Lemma1：用到$p(x)p_S(x_S)=p_{A\cup S}(x_{A\cup S})p_{B\cup S}(x_{B\cup S})$，其中$p_S$指的是子图$\mathcal{G}_S$上的概率测度。由无向图的分解式：$p(x) = \prod\limits_{c\in\mathcal{C}} \psi_c(x)$，因为(A,B,S)分解图$\mathcal{G}$，所有极大团要么属于$A\cup S$要么属于$B\cup S$，所以
$$
p(x)=\prod\limits_{c\in\mathcal{A}}\psi_c(x)\prod\limits_{c\in \mathcal{B}\backslash\mathcal{A}}\psi_c(x)=h(x_{A\cup S})k(x_{B\cup S})​
$$
积分掉B后得到
$$
p_{A\cup S}(x_{A\cup S})=h(x_{A\cup S})\bar{k}(x_S)\\
\bar{k}(x_S)=\int k(x_{B\cup S})\mu_B(dx_B)
$$
对A积分同理，最后得到$p_{A\cup S}(x_{A\cup S})p_{B\cup S}(x_{B\cup S})=p(x)l(x_S)$，积掉A,B就有$l(x_S)=p_S(x_S)$

Hyper Markov Law：A Law $\mathfrak{L} (\theta)$  on $M(\mathscr{G})$ 叫做超马尔可夫律，如果对于$\mathscr{G}$的任意分解$(A,B)$，有
$$
\theta_{A} \perp \theta_{B} | \theta_{A \cap B}​
$$
注意Law实际上是$\theta$的一个概率分布。

strong Hyper Markov Law: A Law $\mathfrak{L} (\theta)$  on $M(\mathscr{G})$叫做强超马尔可夫律，如果对于分解$(A,B)$，有
$$
\theta_{B | A} \perp \theta_{A}
$$

1. non-hierarchical model

$X|\Sigma,g\sim N_p(0,\Sigma)$, $\Sigma|g\sim HIW_g(\alpha, \Phi)$, $p(g)\sim d^{-1}$

$\Phi=\tau(\rho J+ (1-\rho)I)$, $\tau>0$, $\rho \in (-1/(p-1), 1)$

2. hierarchical model


$$
\pi(\tau, \rho) \propto\left[\tau^{p}(1-\rho)^{p-1}\{1+\rho(p-1)\}\right]^{(d-2) / 2}\exp \left\{-\frac{1}{2} \tau\left(\sum_{i=1}^{p} t_{i i}+\rho \sum_{i \neq j} t_{i j}\right)\right\}
$$

$$
\begin{array}{l}{\text { Proposition. Let } \Phi \text { be a random symmetric matrix of form }(3) \text { with } \tau \text { and } \rho \text { distributed as }} \\ {(4), \text { with } \sum_{i \neq j} t_{i j}=0 . \text { Suppose that } d>2-2 / p \text { and let } t_{0}=\sum_{i=1}^{p} t_{i i .} \text { Then }}\end{array}
$$

(a).  $\tau$ and $\rho$ are independent random variables;

(b) $\tau$ has the Gamma distribution  $\tau \sim G a\left(\frac{p(d-2)+2}{2}, \frac{t_{0}}{2}\right)$
$(\mathrm{c})$  $\rho=-\frac{1}{p-1}+\frac{p}{p-1} \gamma$ 

where $\gamma$ has the Beta distribution$\gamma \sim B e\left(\frac{d}{2}, \frac{(p-1)(d-2)+2}{2}\right)$

**Reversible jump MCMC:**

设$\pi(dy)$为目标概率测度，$q_m(y, dy')$为在采取第m种行动时从状态y转移到y'的概率测度，由一般的MH算法得到：
$$
\alpha_{m}\left(y, y^{\prime}\right)=\min \left\{1, \frac{\pi\left(d y^{\prime}\right) q_{m}\left(y^{\prime}, d y\right)}{\pi(d y) q_{m}\left(y, d y^{\prime}\right)}\right\}
$$
对于参数空间维度变化的模型，假设y到y'添加了一个参数u，那么我们假设u与y独立，且有可逆函数h使得$y' = h(y,u)$
$$
\alpha_{m}\left(y, y^{\prime}\right)=\min \left\{1, \frac{\pi\left(y^{\prime}\right)}{\pi(y)} \times \frac{r_{m}\left(y^{\prime}\right)}{r_{m}(y) q(u)} \times\left|\frac{\partial y^{\prime}}{\partial(y, u)}\right|\right\}
$$
$r_m$为采取第m种行动时y或y'的概率，q(u)选择服从0均值的正态分布。

有4种行动

a. 从图g中删去或者添加一条边（随机选中一对(i,j)，如果有边就删除，没有边就添加）

b. 更新不完全矩阵$\Gamma$，和对应的$\Sigma$

c. 更新$\alpha$（$\alpha$是逆威沙特分布的参数，见上）

d. 更新$\Phi $

四种情况下的acceptance rate见paper，很好理解。



## A Monte Carlo Method to Compute the Marginal Likelihood in Non-Decomposable Graphical Gaussian Models

对于$\delta>2, D^{-1}\in M^+(G)$，在$M^{+}(G)$的勒贝格测度下G-Wishart分布可以写为：
$$
f(K | G)=\frac{1}{I_{G}(\delta, D)}|K|^{\frac{\delta-2}{2}} \exp -\frac{1}{2}\langle K, D\rangle
$$
记作$w_{G}(\delta, D)$，$I_{G}(\delta, D)$归一化系数是计算的重点，内积是$tr(AB)$

样本$z^{(1)},..,z^{(n)}$服从多元正态分布，
$$
p(z^{(1)}, ...,z^{(n)}|K,G) = \frac{|K|^{\frac{n}{2}}}{(2 \pi)^{\frac{n p}{2}}} \exp -\frac{1}{2}\langle K, U\rangle​ \\
U = \sum_i^n z^{(i)}z^{(i)\prime}
$$
联合分布：
$$
f \left( Z^{(1)}, \ldots, Z^{(n)}, K, G\right)=\frac{1}{(2 \pi)^{\frac{n p}{2}}|\mathcal{G}|} \frac{1}{I_{G}(\delta, D)}|K|^{\frac{\delta+n-2}{2}} \exp -\frac{1}{2}\langle K, D+U\rangle
$$
那么已知数据，求G的后验分布
$$
\begin{aligned} p\left(G | Z^{(1)}, \ldots, Z^{(n)}\right) &=\frac{J_{G}(\delta, n, D, U)}{\sum_{G^{\prime} \in \mathcal{G}} J_{G^{\prime}}(\delta, n, D, U)} \\ J_{G}(\delta, n, D, U) &=\frac{I_{G}(\delta+n, D+U)}{I_{G}(\delta, D)} \end{aligned}
$$
只需求给定对应G时的归一化系数

- 如果G是完全图($\delta>0, D^{-1}\in M^{+}, a>\frac{p-1}{2}$)

$$
I_{G}(\delta, D)^{-1}=\frac{|D|^{\frac{\delta+p-1}{2}}}{2^{\frac{n p}{2}} \Gamma_{p}\left(\frac{\delta+p-1}{2}\right)}\\
\Gamma_{p}(a)=\pi^{\frac{p(p-1)}{4}} \prod_{i=0}^{i=p-1} \Gamma\left(a-\frac{i}{2}\right)\\
p(\Sigma | \delta, D)=I_{G}(\delta, D)^{-1}|\Sigma|^{-\frac{\delta+2 p}{2}} \exp -\frac{1}{2}\left\langle\Sigma^{-1}, D\right\rangle
$$

- 如果G是可分图

要给出不完全矩阵$\Sigma^{\mathcal{V}}$的概率分布（这个东西。。），即超逆威沙特分布(Hyper Inverse Wishart)，就有
$$
h i w_{G}\left(\Sigma^{\mathcal{V}} | \delta, D\right) d \Sigma^{\mathcal{V}}=\frac{\prod_{j=1}^{j=k} i w\left(\Sigma_{C_{j}} | \delta, D_{C_{j}}\right)}{\prod_{j=2}^{j=k} i w\left(\Sigma_{S_{j}} | \delta, D_{S_{j}}\right)} d \Sigma^{\mathcal{V}}
$$
Roverato(2000)证明了HIW的逆就是定义在$M^{+}(G)$上的$w_G(\delta, D)$，并且
$$
I_{G}(\delta, D)=\frac{\prod_{j=1}^{j=k} I_{G C_{j}}\left(\delta, D_{C_{j}}\right)}{\prod_{j=2}^{j=k} I_{G_{S_{j}}}\left(\delta, D_{S_{j}}\right)}
$$

- 如果G是不可分图

$$
h i w_{G}\left(\Sigma^{\mathcal{V}} | \delta, D\right)=\left(I_{G}(\delta, D)\right)^{-1}|\Sigma|^{-\frac{\delta-2}{2}} J\left(K \mapsto \Sigma^{\mathcal{V}}\right) \exp -\frac{1}{2}\left\langle\Sigma^{-1}, D\right\rangle
$$

唯一的变化就是乘以了一个$J(K\mapsto \Sigma^{\mathcal{V}})$的雅各比行列式，是把$K$的概率密度变成$\Sigma^{\mathcal{V}}$的意思。

这个概率密度也可以分解成prime components和separators的函数，
$$
h i w_{G}\left(\Sigma^{\mathcal{V}} | \delta, D\right) d \Sigma^{\mathcal{V}}=\frac{\prod_{j=1}^{j=k} h i w_{G_{P_{j}}}\left(\Sigma_{P_{j}}^{\mathcal{P}_{j}} | \delta, D_{P_{j}}\right)}{\prod_{j=2}^{j=k} h i w_{G_{S_{j}}}\left(\Sigma_{S_{j}} | \delta, D_{S_{j}}\right)} d \Sigma^{\nu}
$$
$\Sigma_{P_{j}}^{\mathcal{P}_{j}}$代表的是$\mathcal{P}_j$对应的$\Sigma ^{\mathcal{V}}$的子矩阵，对应子图$G_{\mathcal{P}_j}$. 归一化因子又有：
$$
I_{G}(\delta, D)=\frac{\prod_{j=1}^{j=k} I_{G_{P_{j}}}\left(\delta, D_{P_{j}}\right)}{\prod_{j=2}^{j=k} I_{G_{S_{j}}}\left(\delta, D_{S_{j}}\right)}
$$
因为$G_{S_j}$是完全子图，所以可以用上面第一种方法计算分母，分子是对于prime graph的归一化因子的计算，所以整个分析过程中，我们真正需要解决的是这一种情形的计算。

**针对prime graph的蒙特卡洛方法**：

1. 令$K=\phi^T\phi, \phi \in M^{\triangleleft}$，作变换

$$
K \in M^{+}(G) \mapsto \phi^{\mathcal{V}}=\phi_{\mathcal{V}} \in M_{*}^{\triangleleft}(G)
$$

Jacobian为$J_{1}=2^{p} \prod_{i=1}^{i=p} \phi_{i i}^{\nu_{i}+1}$

2. 令$D=\left(T^{t} T\right)^{-1}, T \in M^{\triangleleft}$，作变换

$$
\phi^{\mathcal{V}} \in M_{*}^{\triangleleft}(G) \mapsto \psi^{\mathcal{V}} \in M_{*}^{\triangleleft}(G)
$$

再令$\psi=\phi T^{-1}, \psi^{\mathcal{V}}=\psi_{\mathcal{V}}$，Jacobian为$J_{2}=\prod_{i=1}^{p} t_{i i}^{k_{i}+1}$，$k_i$为the number of vertices preceding i in the given order of the vertices（应该是$\#\{(j,i)\in\mathcal{V}: j<i \}$）

3. 给出$\psi_{ij}, (i,j)\in\bar{\mathcal{V}}$关于$\psi_{ij}, (i,j)\in\mathcal{V}$的表达式

4. $I_G(\delta,D)$表示为$\psi_{ij}, (i,j)\in\mathcal{V}$的函数期望

Lemma（$\psi$和$\phi$的代数关系）：记$t_{\langle js]}=\frac{t_{js}}{t_{ss}}$由线性代数可得
$$
\psi_{s s}=\frac{\phi_{s s}}{t_{s s}}\\
\psi_{r s}=\sum_{j=r}^{s-1}-\psi_{r j} t_{\langle j s]}+\frac{\phi_{r s}}{t_{s s}}\\
$$
${For }(r s) \in \overline{\mathcal{V}} \text { and } r<s$，给出$\psi_{ij}, (i,j)\in\bar{\mathcal{V}}$关于$\psi_{ij}, (i,j)\in\mathcal{V}$的表达式
$$
\begin{array}{c} {\psi_{r s}=\sum_{j=r}^{s-1}-\psi_{r j} t_{\langle j s]}-\sum_{i=1}^{r-1}\left(\frac{\psi_{i r}+\sum_{j=i}^{r-1} \psi_{i j} t_{\langle j r]}}{\psi_{r r}}\right)\left(\psi_{i s}+\sum_{j=i}^{s-1} \psi_{i j} t_{\langle j s]}\right)}\end{array}
$$
上面这行：因为$(rs)\in\bar{\mathcal{V}}$所以$K_{r s}=0=\sum_{i=1}^{i=r} \phi_{i r} \phi_{i s}$，变成$\phi_{r s}=-\sum_{i=1}^{r-1} \frac{\phi_{i r}}{\phi_{r r}} \phi_{i s}$后再代入上上面的两式子就可。

所以由step1
$$
\begin{array}{l}{I_{G}(\delta, D)} \\ {\quad=\int_{M^{+}(G)}|K|^{\frac{\delta-2}{2}} \exp -\frac{1}{2}\langle K, D\rangle d K} \\ 
{\quad=2^{p} \int_{M_{*}^{\triangleleft}(G)} \prod_{i=1}^{i=p}\left(\phi_{i i}^{2}\right)^{\frac{\delta+\nu_{i}-1}{2}} \exp -\frac{1}{2}\left\langle\phi^{t} \phi,\left(T^{t} T\right)^{-1}\right\rangle d \phi^{\mathcal{V}}} \\ 
{\quad=2^{p} \int \prod_{i=1}^{i=p}\left(\phi_{i i}^{2}\right)^{\frac{\delta+\nu_{i}-1}{2}} \exp -\frac{1}{2}\left\langle\left(\phi T^{-1}\right)^{t}, \phi T^{-1}\right\rangle \prod_{i=1}^{i=p} d \phi_{i i} \prod_{i \neq j,(i, j) \in \mathcal{V}} d \phi_{i j}}  \end{array}
$$
第二个等式用了$J_1$和行列式$|K|=|\phi|^2=\prod\phi_{ii}^2$，再经过step2
$$
\begin{array}{l}{I_{G}(\delta, D)} \\ {\quad=2^{p} \prod_{i=1}^{i=p}\left(t_{i i}^{2}\right) \frac{\delta+\nu_{i}-1+k_{i}+1}{2}}\int \prod_{i=1}^{i=p}\left(\psi_{i i}^{2}\right)^{\frac{\delta+\nu_{i}-1}{2}} \exp -\frac{1}{2}\langle \psi^t,\psi \rangle \prod_{i=1}^{i=p} d \psi_{i i} \prod_{(i, j) \in \mathcal{V}, i \neq j} d \psi_{i j} \\ 
{{\quad=2^{p} \prod_{i=1}^{i=p}\left(t_{i i}^{2}\right) \frac{\delta+\nu_{i}-1+k_{i}+1}{2}}\int \prod_{i=1}^{i=p}\left(\psi_{i i}^{2}\right)^{\frac{\delta+\nu_{i}-1}{2}} \exp -\frac{1}{2}\left(\sum_{i=1}^{i=p} \psi_{i i}^{2}+\sum_{(i, j) \in \mathcal{V}, i \neq j} \psi_{i j}^{2}+\sum_{(i, j) \in \overline{\mathcal{V}}} \psi_{i j}^{2}\right) \prod_{i=1}^{i=p} d \psi_{i i} \prod_{(i, j) \in \mathcal{V}, i \neq j} d \psi_{i j}} \end{array}
$$
令$b_i=\nu_i+k_i+1$，那么，
$$
\begin{array}{l} {I_{G}(\delta, D) \\ {= 2^{p}  \prod_{i=1}^{i=p}\left(t_{i i}^{2}\right)^{\frac{\delta+b_{i}-1}{2}} \int \exp -\frac{1}{2} \sum_{\overline{V}} \psi_{i j}^{2} \prod_{i=1}^{i=p}\left(\psi_{i j}^{2}\right)^{\frac{\delta+\nu_{i}-1}{2}} \exp -\frac{1}{2} \sum_{i=1}^{i=p} \psi_{i i}^{2}}} \\ {\qquad\times\exp -\frac{1}{2} \sum_{(i, j) \in \mathcal{V}, i \neq j} \psi_{i j}^{2} \prod_{i=1}^{i=p} d \psi_{i i} \prod_{(i, j) \in \mathcal{V}, i \neq j} d \psi_{i j}} \end{array}
$$
最后一次代换$d \psi_{i i}=\frac{1}{2} \psi_{i i}^{-1} d\left(\psi_{i i}^{2}\right)$
$$
I_{G}(\delta, D)=\prod_{i=1}^{i=p} 2^{\frac{\delta+\nu_{i}}{2}}(2 \pi)^{\frac{\nu_{i}}{2}} \Gamma\left(\frac{\delta+\nu_{i}}{2}\right) \prod_{i=1}^{i=p}\left(t_{i i}^{2}\right)^{\frac{\delta+b_{i}-1}{2}} \\
\times \int \exp -\frac{1}{2} \sum_{(i, j) \in \overline{\mathcal{V}}} \psi_{i j}^{2} \prod_{i=1}^{i=p} \frac{1}{\Gamma\left(\frac{\delta+\nu_{i}}{2}\right)}\left(\frac{\psi_{i i}^{2}}{2}\right)^{\frac{\delta+\nu_{i}}{2}-1} \exp -\frac{1}{2} \psi_{i i}^{2} \\ \times
\prod_{(i, j) \in \mathcal{V}, i \neq j} \frac{1}{\sqrt{2 \pi}} \exp -\frac{1}{2} \psi_{i j}^{2} \quad \prod_{i=1}^{i=p} d\left(\psi_{i i}\right)^{2} \prod_{(i, j) \in \mathcal{V}, i \neq j} d \psi_{i j}
$$
重点关注积分号后面的表达式，$\frac{1}{\Gamma\left(\frac{\delta+\nu_{i}}{2}\right)}\left(\frac{\psi_{i i}^{2}}{2}\right)^{\frac{\delta+\nu_{i}}{2}-1} \exp -\frac{1}{2} \psi_{i i}^{2}$是$\chi^2(\delta+\nu_i)$的概率密度，$\frac{1}{\sqrt{2 \pi}} \exp -\frac{1}{2} \psi_{i j}^{2}$是标准正态的分布函数，所以非常巧妙的，积分刚好是求$-\frac{1}{2} \sum_{(i, j) \in \overline{\mathcal{V}}} \psi_{i j}^{2}$的期望！（只需要我们抽样的时候$\psi$是从这两个分布里取的就行）

具体实验设计就应该能看懂了