"""PageRank analysis of graph structure. """
from warnings import warn

import networkx as nx

# @not_implemented_for('multigraph')
def pagerank(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
# G是图，alpha随机游走参数
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specfiied, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = nx.pagerank(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop after
    an error tolerance of ``len(G) * tol`` has been reached. If the
    number of iterations exceed `max_iter`, a
    :exc:`networkx.exception.PowerIterationFailedConvergence` exception
    is raised.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.

    See Also
    --------
    pagerank_numpy, pagerank_scipy, google_matrix

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf

    """
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
	# 目前D.adj格式为dict,每一组键值对是
	# source point:{destinatio1:{},destination2:{},...}
	
    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    # W中统计了每一个destination的权重
    # source point:{destinatio1:{1/该源点的出度},destination2:{1/该源点的出度},...}
    # 例如source=‘3’，destinations={‘1’，‘2’}
    # 则存在键值对
    # '3':{'1':{'weight':0.5},'2':{'weight':0.5}}
    N = W.number_of_nodes()# 点的总数

    # Choose fixed starting vector if not given
    if nstart is None:
    	#  fromkeys() 函数用于创建一个新字典，以序列 seq 中元素做字典的键，value 为字典所有键对应的初始值
    	# 点：1/点的个数，所有value均相同
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    # W.outdegree={(source:出度)}
    # dangling_nodes为出度=0的点
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for i in range(max_iter): # max_iter=100
        xlast = x
        # x的所有键对应的值清零，fromkeys返回一个新字典，不修改原变量
        x = dict.fromkeys(xlast.keys(), 0)
        # 所有出度为0的点求和 alpha*（旧迭代向量r_oldr中出度为0的部分对应值的和）
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x: # n是键,eg. '30'
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]: # W[‘30’]，from=‘30’对应的to点，{to1:{'weight':1/'30'出度},to2:{..}...}
            	# 基础值
                # print('W[n]',n, nbr, W[n])
                # print('W[n][nbr],',n, nbr, W[n][nbr])
                # x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
                x[nbr] += alpha * xlast[n] * W[n][nbr][0][weight]
			
			# 加入随机游走的值(1.0 - alpha) * p.get(n, 0)
			# get函数：返回dangling_weights中键为n的值，如没有则返回1
			# 若使用默认参数，则p=dangling_weights为{点:1/node_num}（所有存在的点）
			# danglesum * dangling_weights.get(n, 0)=danglesum/node_num=r_old中所有出度为0的点对应value的和 * 随机游走概率 / 所有点的个数 加这个是为了让列的和为1
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    print('iterations', i)
    raise nx.PowerIterationFailedConvergence(max_iter)

