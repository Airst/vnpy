import React, { useState, useEffect, useRef } from 'react';
import { Card, Select, Spin, Empty, Tag, Typography, Table, Modal, Row, Col, Divider, Alert, message, Button, Popconfirm, Progress } from 'antd';
import { FileTextOutlined, CheckCircleOutlined, CloseCircleOutlined, MinusCircleOutlined, EyeOutlined, ReloadOutlined, LoadingOutlined, CloudUploadOutlined, DeleteOutlined } from '@ant-design/icons';

const { Text, Title, Paragraph } = Typography;

const LlmEvaluation = () => {
    const [loading, setLoading] = useState(false);
    const [ratings, setRatings] = useState([]);
    const [total, setTotal] = useState(0);
    const [totalUnfiltered, setTotalUnfiltered] = useState(0);
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(20);
    const [summary, setSummary] = useState({ good: 0, bad: 0, neutral: 0, error: 0, avg_confidence: 'N/A' });
    const [filterRating, setFilterRating] = useState(null);
    const [detailModal, setDetailModal] = useState({ visible: false, rating: null });
    const [taskStatus, setTaskStatus] = useState(null);
    const [batchStatus, setBatchStatus] = useState(null);
    const pollTimerRef = useRef(null);
    const batchPollTimerRef = useRef(null);
    // Refs to always get latest state values in callbacks
    const filterRatingRef = useRef(filterRating);
    const pageRef = useRef(page);
    const pageSizeRef = useRef(pageSize);

    // Signal selection
    const [signals, setSignals] = useState([]);
    const [selectedSignal, setSelectedSignal] = useState('ashare_mlp_signal_v9');
    const [signalDate, setSignalDate] = useState(null);

    // Keep refs updated
    useEffect(() => { filterRatingRef.current = filterRating; }, [filterRating]);
    useEffect(() => { pageRef.current = page; }, [page]);
    useEffect(() => { pageSizeRef.current = pageSize; }, [pageSize]);

    // Stock search related state
    const [searchOptions, setSearchOptions] = useState([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const searchTimeoutRef = useRef(null);
    const [searchedSymbol, setSearchedSymbol] = useState(null);
    const [isSearchMode, setIsSearchMode] = useState(false);

    // Toggle filter when clicking summary tags
    const handleFilterClick = (ratingType) => {
        const newFilter = filterRating === ratingType ? null : ratingType;
        setFilterRating(newFilter);
        setPage(1);
        loadRatings(1, pageSizeRef.current, newFilter);
    };

    // Load all ratings with pagination
    const loadRatings = async (pageNum, ps, filter) => {
        setLoading(true);
        try {
            const params = new URLSearchParams({
                page: pageNum,
                page_size: ps,
            });
            if (filter) {
                params.append('rating_filter', filter);
            }
            if (selectedSignal) {
                params.append('signal_name', selectedSignal);
            }
            const res = await fetch(`/api/llm_ratings/all?${params}`);
            const data = await res.json();
            setRatings(data.ratings || []);
            setTotal(data.total || 0);
            if (data.total_unfiltered !== undefined) {
                setTotalUnfiltered(data.total_unfiltered);
            } else {
                setTotalUnfiltered(data.total || 0);
            }
            setPage(data.page || pageNum);
            if (data.signal_date) {
                setSignalDate(data.signal_date);
            }
            if (data.summary) {
                setSummary(data.summary);
            }
        } catch (error) {
            console.error("Failed to load ratings", error);
        } finally {
            setLoading(false);
        }
    };

    // Load on mount
    useEffect(() => {
        loadSignals();
        loadRatings(1, pageSize, null);
    }, []);

    // Load available signals
    const loadSignals = async () => {
        try {
            const res = await fetch('/api/signals');
            const data = await res.json();
            if (data.signals && data.signals.length > 0) {
                setSignals(data.signals);
                // Default to v9 signal if available
                const v9Signal = data.signals.find(s => s.includes('v9'));
                if (v9Signal) {
                    setSelectedSignal(v9Signal);
                } else {
                    setSelectedSignal(data.signals[0]);
                }
            }
        } catch (error) {
            console.error("Failed to load signals", error);
        }
    };

    // Cleanup poll timers on unmount
    useEffect(() => {
        return () => {
            if (pollTimerRef.current) clearInterval(pollTimerRef.current);
            if (batchPollTimerRef.current) clearInterval(batchPollTimerRef.current);
            if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
        };
    }, []);

    // Refresh list
    const handleRefreshList = () => {
        if (isSearchMode && searchedSymbol) {
            handleStockSelect(searchedSymbol);
        } else {
            loadRatings(pageRef.current, pageSizeRef.current, filterRatingRef.current);
        }
    };

    // Handle page change
    const handlePageChange = (pageNum, ps) => {
        setPage(pageNum);
        setPageSize(ps);
        loadRatings(pageNum, ps, filterRatingRef.current);
    };

    // Delete a rating
    const handleDelete = async (vt_symbol, date) => {
        try {
            const url = `/api/llm_ratings/stock/${vt_symbol}${date ? `?date=${date}` : ''}`;
            const res = await fetch(url, { method: 'DELETE' });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || '删除失败');
            }
            message.success(`${vt_symbol} 评估记录已删除`);
            // Refresh current view
            if (isSearchMode && searchedSymbol) {
                handleStockSelect(searchedSymbol);
            } else {
                loadRatings(page, pageSize, filterRating);
            }
        } catch (error) {
            message.error(`删除失败: ${error.message}`);
        }
    };

    // Stock search handler
    const handleStockSearch = (value) => {
        if (searchTimeoutRef.current) {
            clearTimeout(searchTimeoutRef.current);
        }
        
        if (!value || value.length < 2) {
            setSearchOptions([]);
            return;
        }

        searchTimeoutRef.current = setTimeout(() => {
            setSearchLoading(true);
            fetch(`/api/symbols/search?keyword=${value}`)
                .then(res => res.json())
                .then(data => {
                    if (data.symbols) {
                        setSearchOptions(data.symbols);
                    }
                    setSearchLoading(false);
                })
                .catch(err => {
                    console.error("Search failed", err);
                    setSearchLoading(false);
                });
        }, 500);
    };

    // Handle stock selection from search
    const handleStockSelect = async (value) => {
        if (!value) {
            setSearchedSymbol(null);
            setIsSearchMode(false);
            loadRatings(1, pageSize, filterRating);
            return;
        }

        setSearchedSymbol(value);
        setIsSearchMode(true);
        setFilterRating(null);

        setLoading(true);
        try {
            const res = await fetch(`/api/llm_ratings/stock/${value}`);
            if (res.ok) {
                const data = await res.json();
                setRatings(data.history || []);
                setTotal(data.history?.length || 0);
            } else {
                setRatings([{
                    vt_symbol: value,
                    rating: null,
                    reason: '该股票尚未被 LLM 评估',
                    confidence: 0,
                    target_direction: null,
                    analysis_dimensions: {},
                    key_factors: [],
                    error: null,
                    not_evaluated: true,
                }]);
                setTotal(1);
            }
        } catch (error) {
            console.error('Failed to load stock rating', error);
            setRatings([{
                vt_symbol: value,
                rating: null,
                reason: '该股票尚未被 LLM 评估',
                confidence: 0,
                target_direction: null,
                analysis_dimensions: {},
                key_factors: [],
                error: null,
                not_evaluated: true,
            }]);
            setTotal(1);
        } finally {
            setLoading(false);
        }
    };

    // Poll task status when running
    const startPolling = (vt_symbol) => {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        
        pollTimerRef.current = setInterval(async () => {
            try {
                const res = await fetch(`/api/llm_ratings/task_status/${vt_symbol}`);
                const data = await res.json();
                setTaskStatus(data);
                
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollTimerRef.current);
                    pollTimerRef.current = null;
                    if (isSearchMode && searchedSymbol) {
                        handleStockSelect(searchedSymbol);
                    } else {
                        loadRatings(pageRef.current, pageSizeRef.current, filterRatingRef.current);
                    }
                    
                    if (data.status === 'completed') {
                        message.success(`${vt_symbol} 评估完成`);
                    }
                }
            } catch (error) {
                console.error('Failed to poll task status', error);
            }
        }, 3000);
    };

    // Trigger re-evaluate
    const handleReevaluate = async (vt_symbol, score) => {
        setTaskStatus({ status: 'running', message: 'LLM 评估中...' });
        
        try {
            const url = `/api/llm_ratings/reevaluate?vt_symbol=${vt_symbol}&score=${score || 0}`;
            const res = await fetch(url, { method: 'POST' });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || '提交失败');
            }
            
            message.success(`${vt_symbol} 评估任务已提交，后台执行中...`);
            startPolling(vt_symbol);
        } catch (error) {
            setTaskStatus({ status: 'failed', message: error.message });
            message.error(`提交失败: ${error.message}`);
        }
    };

    // Batch re-evaluate failed stocks
    const startBatchPolling = () => {
        if (batchPollTimerRef.current) clearInterval(batchPollTimerRef.current);
        
        batchPollTimerRef.current = setInterval(async () => {
            try {
                const res = await fetch('/api/llm_ratings/batch_status');
                const data = await res.json();
                setBatchStatus(data);
                
                if (!data.running) {
                    clearInterval(batchPollTimerRef.current);
                    batchPollTimerRef.current = null;
                    if (isSearchMode && searchedSymbol) {
                        handleStockSelect(searchedSymbol);
                    } else {
                        loadRatings(pageRef.current, pageSizeRef.current, filterRatingRef.current);
                    }
                    
                    if (data.completed > 0) {
                        message.success(`批量评估完成：${data.completed} 成功，${data.failed} 失败`);
                    }
                }
            } catch (error) {
                console.error('Failed to poll batch status', error);
            }
        }, 5000);
    };

    const handleBatchReevaluate = async () => {
        setBatchStatus({ running: true, total: 0, completed: 0, failed: 0, results: [] });
        
        try {
            const res = await fetch('/api/llm_ratings/reevaluate_failed', { method: 'POST' });
            const data = await res.json();
            
            if (data.count === 0) {
                message.info('没有评估失败的股票');
                setBatchStatus(null);
                return;
            }
            
            message.success(`批量评估任务已提交，共 ${data.count} 只股票`);
            setBatchStatus({ running: true, total: data.count, completed: 0, failed: 0, results: [] });
            startBatchPolling();
        } catch (error) {
            setBatchStatus(null);
            message.error(`提交失败: ${error.message}`);
        }
    };

    const getRatingTag = (rating) => {
        const config = {
            good: { color: 'green', icon: <CheckCircleOutlined />, text: 'Good' },
            bad: { color: 'red', icon: <CloseCircleOutlined />, text: 'Bad' },
            neutral: { color: 'default', icon: <MinusCircleOutlined />, text: 'Neutral' },
        };
        const c = config[rating?.toLowerCase()] || config.neutral;
        return <Tag color={c.color} icon={c.icon}>{c.text}</Tag>;
    };

    const getDirectionTag = (direction) => {
        const config = {
            up: { color: 'green', text: '看涨' },
            down: { color: 'red', text: '看跌' },
            flat: { color: 'default', text: '震荡' },
        };
        const c = config[direction?.toLowerCase()] || config.flat;
        return <Tag color={c.color}>{c.text}</Tag>;
    };

    const renderSummary = () => {
        if (isSearchMode) return null;

        const { good, bad, neutral, avg_confidence } = summary;
        const filterActive = (type) => filterRating === type;

        return (
            <Card title="评估概览" bordered={false} style={{ marginBottom: 20 }}>
                <Row gutter={[16, 16]}>
                    <Col span={6}>
                        <div style={{ textAlign: 'center' }}>
                            <Text type="secondary">评估股票数</Text>
                            <div style={{ marginTop: 8 }}><Text strong>{totalUnfiltered}</Text></div>
                        </div>
                    </Col>
                    <Col span={4}>
                        <div style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleFilterClick('good')}>
                            <Text type="secondary">Good (看好)</Text>
                            <div style={{ marginTop: 8 }}>
                                <Tag color={filterActive('good') ? 'blue' : 'green'} style={{ transition: 'all 0.3s' }}>{good}</Tag>
                            </div>
                        </div>
                    </Col>
                    <Col span={4}>
                        <div style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleFilterClick('bad')}>
                            <Text type="secondary">Bad (看空)</Text>
                            <div style={{ marginTop: 8 }}>
                                <Tag color={filterActive('bad') ? 'blue' : 'red'} style={{ transition: 'all 0.3s' }}>{bad}</Tag>
                            </div>
                        </div>
                    </Col>
                    <Col span={4}>
                        <div style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleFilterClick('neutral')}>
                            <Text type="secondary">Neutral (中性)</Text>
                            <div style={{ marginTop: 8 }}>
                                <Tag color={filterActive('neutral') ? 'blue' : 'default'} style={{ transition: 'all 0.3s' }}>{neutral}</Tag>
                            </div>
                        </div>
                    </Col>
                    <Col span={6}>
                        <div style={{ textAlign: 'center' }}>
                            <Text type="secondary">平均置信度</Text>
                            <div style={{ marginTop: 8 }}><Text strong>{avg_confidence}</Text></div>
                        </div>
                    </Col>
                </Row>
            </Card>
        );
    };

    const handleOpenDetail = () => {
        setTaskStatus(null);
        if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
        }
    };

    const handleCloseDetail = () => {
        setDetailModal({ visible: false, rating: null });
        setTaskStatus(null);
        if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
        }
    };

    const handleRefreshDetail = async (vt_symbol) => {
        setTaskStatus(null);
        try {
            const res = await fetch(`/api/llm_ratings/stock/${vt_symbol}`);
            if (res.ok) {
                const data = await res.json();
                const latest = data.history[data.history.length - 1];
                setDetailModal({ visible: true, rating: latest });
                message.success(`${vt_symbol} 数据已刷新`);
            }
        } catch (error) {
            console.error('Failed to refresh detail', error);
        }
    };

    const renderDetailModal = () => {
        const { visible, rating } = detailModal;
        if (!rating) return null;

        // Handle not-evaluated stock
        if (rating.not_evaluated) {
            return (
                <Modal
                    title={`${rating.vt_symbol} - 评估详情`}
                    open={visible}
                    onCancel={handleCloseDetail}
                    afterOpenChange={(open) => open && handleOpenDetail()}
                    footer={[
                        <Button 
                            key="evaluate" 
                            icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : <ReloadOutlined />}
                            onClick={() => handleReevaluate(rating.vt_symbol, 0)}
                            loading={taskStatus?.status === 'running'}
                            disabled={taskStatus?.status === 'running'}
                            type="primary"
                        >
                            {taskStatus?.status === 'running' ? '评估中...' : 
                             taskStatus?.status === 'completed' ? '评估完成' :
                             taskStatus?.status === 'failed' ? '重试' : '开始评估'}
                        </Button>,
                        <Button 
                            key="refresh" 
                            icon={<ReloadOutlined />}
                            onClick={() => handleRefreshDetail(rating.vt_symbol)}
                        >
                            刷新
                        </Button>,
                        <Button key="close" onClick={handleCloseDetail}>关闭</Button>
                    ]}
                    width={800}
                >
                    {taskStatus && (
                        <Alert
                            message={taskStatus.status === 'running' ? '任务执行中' : 
                                     taskStatus.status === 'completed' ? '评估完成' : '评估失败'}
                            description={taskStatus.message}
                            type={taskStatus.status === 'running' ? 'info' : 
                                  taskStatus.status === 'completed' ? 'success' : 'error'}
                            showIcon
                            icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : undefined}
                            style={{ marginBottom: 16 }}
                        />
                    )}
                    <Empty description={rating.reason} />
                </Modal>
            );
        }

        const dimensions = rating.analysis_dimensions || {};
        const keyFactors = rating.key_factors || [];
        const positiveFactors = keyFactors.filter(f => f.type === 'positive');
        const negativeFactors = keyFactors.filter(f => f.type === 'negative');

        return (
            <Modal
                title={`${rating.vt_symbol} - 评估详情`}
                open={visible}
                onCancel={handleCloseDetail}
                afterOpenChange={(open) => open && handleOpenDetail()}
                footer={[
                    <Button 
                        key="reevaluate" 
                        icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : <ReloadOutlined />}
                        onClick={() => handleReevaluate(rating.vt_symbol, rating.score)}
                        loading={taskStatus?.status === 'running'}
                        disabled={taskStatus?.status === 'running'}
                    >
                        {taskStatus?.status === 'running' ? '评估中...' : 
                         taskStatus?.status === 'completed' ? '重新评估' :
                         taskStatus?.status === 'failed' ? '重试' : '重新评估'}
                    </Button>,
                    <Button 
                        key="refresh" 
                        icon={<ReloadOutlined />}
                        onClick={() => handleRefreshDetail(rating.vt_symbol)}
                    >
                        刷新
                    </Button>,
                    <Button key="close" type="primary" onClick={handleCloseDetail}>关闭</Button>
                ]}
                width={800}
            >
                {taskStatus && (
                    <Alert
                        message={taskStatus.status === 'running' ? '任务执行中' : 
                                 taskStatus.status === 'completed' ? '评估完成' : '评估失败'}
                        description={taskStatus.message}
                        type={taskStatus.status === 'running' ? 'info' : 
                              taskStatus.status === 'completed' ? 'success' : 'error'}
                        showIcon
                        icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : undefined}
                        style={{ marginBottom: 16 }}
                    />
                )}
                {rating.date && (
                    <div style={{ marginBottom: 12 }}>
                        <Text type="secondary">评估日期：</Text>
                        <Text strong>{rating.date}</Text>
                    </div>
                )}
                <div style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 16]}>
                        <Col span={8}>
                            <Text type="secondary">评级：</Text>
                            {getRatingTag(rating.rating)}
                        </Col>
                        <Col span={8}>
                            <Text type="secondary">预测方向：</Text>
                            {getDirectionTag(rating.target_direction)}
                        </Col>
                        <Col span={8}>
                            <Text type="secondary">置信度：</Text>
                            <Text strong>{(rating.confidence * 100).toFixed(0)}%</Text>
                        </Col>
                    </Row>
                </div>

                {rating.score !== undefined && rating.score !== null && (
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">模型分数：</Text>
                        <Text code>{rating.score?.toFixed(4)}</Text>
                    </div>
                )}

                {rating.stop_loss_price && (
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">止损价：</Text>
                        <Text code>{rating.stop_loss_price}</Text>
                    </div>
                )}

                <div style={{ marginBottom: 16 }}>
                    <Text type="secondary">评级理由：</Text>
                    <Paragraph style={{ marginTop: 4 }}>{rating.reason}</Paragraph>
                </div>

                {Object.keys(dimensions).length > 0 && (
                    <>
                        <Divider orientation="left">分析维度</Divider>
                        <Row gutter={[8, 12]}>
                            {dimensions.technical && (
                                <Col span={12}>
                                    <Text type="secondary">技术面：</Text>
                                    <div>{dimensions.technical}</div>
                                </Col>
                            )}
                            {dimensions.fundamental && (
                                <Col span={12}>
                                    <Text type="secondary">基本面：</Text>
                                    <div>{dimensions.fundamental}</div>
                                </Col>
                            )}
                            {dimensions.event && (
                                <Col span={12}>
                                    <Text type="secondary">事件/催化剂：</Text>
                                    <div>{dimensions.event}</div>
                                </Col>
                            )}
                            {dimensions.sentiment && (
                                <Col span={12}>
                                    <Text type="secondary">市场情绪：</Text>
                                    <div>{dimensions.sentiment}</div>
                                </Col>
                            )}
                        </Row>
                    </>
                )}

                {(positiveFactors.length > 0 || negativeFactors.length > 0) && (
                    <>
                        <Divider orientation="left">关键因素</Divider>
                        <Row gutter={16}>
                            {positiveFactors.length > 0 && (
                                <Col span={12}>
                                    <Text type="success">看多因素：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {positiveFactors.map((f, i) => (
                                            <li key={i}><Text>{f.content}</Text></li>
                                        ))}
                                    </ul>
                                </Col>
                            )}
                            {negativeFactors.length > 0 && (
                                <Col span={12}>
                                    <Text type="danger">看空因素：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {negativeFactors.map((f, i) => (
                                            <li key={i}><Text>{f.content}</Text></li>
                                        ))}
                                    </ul>
                                </Col>
                            )}
                        </Row>
                    </>
                )}

                {rating.error && (
                    <Alert
                        message="LLM 调用错误"
                        description={rating.error}
                        type="error"
                        showIcon
                        style={{ marginTop: 12 }}
                    />
                )}
            </Modal>
        );
    };

    // Table columns
    const columns = [
        {
            title: '股票代码',
            dataIndex: 'vt_symbol',
            key: 'vt_symbol',
            width: 130,
            fixed: 'left',
        },
        {
            title: '评估日期',
            dataIndex: 'date',
            key: 'date',
            width: 120,
        },
        {
            title: '评级',
            dataIndex: 'rating',
            key: 'rating',
            width: 100,
            render: (rating) => rating ? getRatingTag(rating) : '-',
        },
        {
            title: '预测方向',
            dataIndex: 'target_direction',
            key: 'target_direction',
            width: 100,
            render: (direction) => direction ? getDirectionTag(direction) : '-',
        },
        {
            title: '置信度',
            dataIndex: 'confidence',
            key: 'confidence',
            width: 100,
            render: (conf) => conf ? `${(conf * 100).toFixed(0)}%` : '-',
            sorter: (a, b) => (a.confidence || 0) - (b.confidence || 0),
        },
        {
            title: '模型分数',
            dataIndex: 'score',
            key: 'score',
            width: 120,
            render: (score) => score !== null && score !== undefined ? score.toFixed(4) : '-',
        },
        {
            title: '评级理由',
            dataIndex: 'reason',
            key: 'reason',
            ellipsis: true,
        },
        {
            title: '操作',
            key: 'action',
            width: 140,
            fixed: 'right',
            render: (_, record) => (
                <>
                    <Button 
                        type="link" 
                        icon={<EyeOutlined />} 
                        onClick={() => setDetailModal({ visible: true, rating: record })}
                    >
                        详情
                    </Button>
                    <Popconfirm
                        title="删除评估记录"
                        description={`确定删除 ${record.vt_symbol} 的评估记录吗？`}
                        onConfirm={() => handleDelete(record.vt_symbol, record.date)}
                        okText="删除"
                        cancelText="取消"
                    >
                        <Button type="link" danger icon={<DeleteOutlined />}>
                            删除
                        </Button>
                    </Popconfirm>
                </>
            ),
        },
    ];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Search and batch actions */}
            <Card title="LLM 评估" bordered={false}
                extra={
                    <Popconfirm
                        title="重新评估所有失败股票"
                        description="将对所有评估失败的股票重新执行 LLM 评估，是否继续？"
                        onConfirm={handleBatchReevaluate}
                        okText="确认"
                        cancelText="取消"
                        disabled={batchStatus?.running}
                    >
                        <Button 
                            icon={batchStatus?.running ? <LoadingOutlined spin /> : <CloudUploadOutlined />}
                            loading={batchStatus?.running}
                            disabled={batchStatus?.running}
                        >
                            {batchStatus?.running ? `批量评估中 (${batchStatus.completed}/${batchStatus.total})` : '批量重评失败股票'}
                        </Button>
                    </Popconfirm>
                }
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <FileTextOutlined style={{ fontSize: 20 }} />
                    <Select
                        style={{ width: 250 }}
                        placeholder="选择模型信号"
                        options={signals.map(s => ({ value: s, label: s }))}
                        value={selectedSignal}
                        onChange={(value) => {
                            setSelectedSignal(value);
                            // Reload ratings with new signal scores
                            setFilterRating(null);
                            setPage(1);
                            loadRatings(1, pageSizeRef.current, null);
                        }}
                    />
                    <Select
                        style={{ flex: 1 }}
                        placeholder="搜索股票代码（代码或名称）"
                        options={searchOptions}
                        onChange={handleStockSelect}
                        onSearch={handleStockSearch}
                        filterOption={false}
                        notFoundContent={searchLoading ? <Spin size="small" /> : null}
                        showSearch
                        allowClear
                        value={searchedSymbol}
                    />
                </div>

                {signalDate && (
                    <div style={{ marginTop: 8 }}>
                        <Text type="secondary">信号日期：{signalDate}</Text>
                    </div>
                )}

                {/* Batch task progress */}
                {batchStatus?.running && (
                    <div style={{ marginTop: 16 }}>
                        <Progress 
                            percent={batchStatus.total > 0 ? Math.round((batchStatus.completed / batchStatus.total) * 100) : 0}
                            status="active"
                            format={() => `已完成 ${batchStatus.completed}/${batchStatus.total}，失败 ${batchStatus.failed}`}
                        />
                    </div>
                )}
                {batchStatus && !batchStatus.running && batchStatus.completed > 0 && (
                    <Alert
                        message="批量评估完成"
                        description={`${batchStatus.completed} 只成功，${batchStatus.failed} 只失败`}
                        type={batchStatus.failed === 0 ? 'success' : 'warning'}
                        showIcon
                        closable
                        onClose={() => setBatchStatus(null)}
                        style={{ marginTop: 16 }}
                    />
                )}
            </Card>

            {/* Data display area */}
            <Spin spinning={loading}>
                {ratings.length > 0 || isSearchMode || filterRating ? (
                    <>
                        {renderSummary()}
                        {filterRating && !isSearchMode && (
                            <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                                <Tag 
                                    color="blue" 
                                    closable 
                                    onClose={() => { setFilterRating(null); setPage(1); loadRatings(1, pageSizeRef.current, null); }}
                                    style={{ fontSize: 14, padding: '4px 12px' }}
                                >
                                    当前筛选: {filterRating === 'good' ? 'Good (看好)' : filterRating === 'bad' ? 'Bad (看空)' : 'Neutral (中性)'}
                                </Tag>
                                <Button 
                                    size="small" 
                                    icon={<ReloadOutlined />} 
                                    onClick={() => { setFilterRating(null); setPage(1); loadRatings(1, pageSizeRef.current, null); }}
                                >
                                    取消筛选
                                </Button>
                            </div>
                        )}
                        <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'flex-end' }}>
                            <Button 
                                size="small" 
                                icon={<ReloadOutlined />} 
                                onClick={handleRefreshList}
                            >
                                刷新列表
                            </Button>
                        </div>
                        <Table
                            columns={columns}
                            dataSource={isSearchMode ? ratings : ratings}
                            rowKey={(record) => `${record.vt_symbol}_${record.date || 'latest'}`}
                            pagination={isSearchMode ? false : {
                                current: page,
                                pageSize: pageSize,
                                total: total,
                                showSizeChanger: true,
                                showTotal: (t) => `共 ${t} 条`,
                                onChange: handlePageChange,
                                onShowSizeChange: handlePageChange,
                            }}
                            scroll={{ x: 1100 }}
                            size="middle"
                        />
                        {renderDetailModal()}
                    </>
                ) : (
                    <Empty description="请搜索股票代码查看评估详情" />
                )}
            </Spin>
        </div>
    );
};

export default LlmEvaluation;
