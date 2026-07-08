import React, { useState, useEffect, useRef } from 'react';
import { Card, Select, Spin, Empty, Tag, Typography, Table, Row, Col, Alert, message, Button, Popconfirm, Progress } from 'antd';
import { FileTextOutlined, CheckCircleOutlined, CloseCircleOutlined, MinusCircleOutlined, EyeOutlined, ReloadOutlined, LoadingOutlined, CloudUploadOutlined, DeleteOutlined, LineChartOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import StockRatingDetailModal from './StockRatingDetailModal';

const { Text } = Typography;

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
    const [batchStatus, setBatchStatus] = useState(null);
    const batchPollTimerRef = useRef(null);
    // Refs to always get latest state values in callbacks
    const filterRatingRef = useRef(filterRating);
    const pageRef = useRef(page);
    const pageSizeRef = useRef(pageSize);
    const selectedSignalRef = useRef(null);

    // Signal selection
    const [signals, setSignals] = useState([]);
    const [selectedSignal, setSelectedSignal] = useState(null);
    const [signalDate, setSignalDate] = useState(null);

    // Stock search related state
    const [searchOptions, setSearchOptions] = useState([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const searchTimeoutRef = useRef(null);
    const [searchedSymbol, setSearchedSymbol] = useState(null);
    const [isSearchMode, setIsSearchMode] = useState(false);

    // Keep refs updated
    useEffect(() => { filterRatingRef.current = filterRating; }, [filterRating]);
    useEffect(() => { pageRef.current = page; }, [page]);
    useEffect(() => { pageSizeRef.current = pageSize; }, [pageSize]);
    useEffect(() => { selectedSignalRef.current = selectedSignal; }, [selectedSignal]);

    // Toggle filter when clicking summary tags
    const handleFilterClick = (ratingType) => {
        const newFilter = filterRating === ratingType ? null : ratingType;
        setFilterRating(newFilter);
        setPage(1);
        loadRatings(1, pageSizeRef.current, newFilter);
    };

    // Load all ratings with pagination
    const loadRatings = async (pageNum, ps, filter, signalName) => {
        setLoading(true);
        try {
            const params = new URLSearchParams({
                page: pageNum,
                page_size: ps,
            });
            if (filter) {
                params.append('rating_filter', filter);
            }
            const sig = signalName ?? selectedSignal;
            if (sig) {
                params.append('signal_name', sig);
            }
            const res = await fetch(`/api/llm_ratings?${params}`);
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
                // Default to the latest (last) signal
                setSelectedSignal(data.signals[data.signals.length - 1]);
            }
        } catch (error) {
            console.error("Failed to load signals", error);
        }
    };

    // Cleanup poll timers on unmount
    useEffect(() => {
        return () => {
            if (batchPollTimerRef.current) clearInterval(batchPollTimerRef.current);
            if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
        };
    }, []);

    // Refresh list
    const handleRefreshList = () => {
        if (isSearchMode && searchedSymbol) {
            handleStockSelect(searchedSymbol);
        } else {
            loadRatings(pageRef.current, pageSizeRef.current, filterRatingRef.current, selectedSignalRef.current);
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
            const params = new URLSearchParams({
                vt_symbol: value,
            });
            if (selectedSignal) {
                params.append('signal_name', selectedSignal);
            }
            const res = await fetch(`/api/llm_ratings?${params}`);
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
            const params = new URLSearchParams();
            if (selectedSignal) {
                params.append('signal_name', selectedSignal);
            }
            const url = `/api/llm_ratings/reevaluate_failed?${params}`;
            const res = await fetch(url, { method: 'POST' });
            const data = await res.json();
            
            if (data.count === 0) {
                message.info('没有需要评估的股票');
                setBatchStatus(null);
                return;
            }
            
            message.success(`批量评估任务已提交，共 ${data.count} 只股票（失败 ${data.failed} + 未评估 ${data.unrated}）`);
            setBatchStatus({ running: true, total: data.count, completed: 0, failed: 0, results: [] });
            startBatchPolling();
        } catch (error) {
            setBatchStatus(null);
            message.error(`提交失败: ${error.message}`);
        }
    };

    const handleReevaluateAll = async () => {
        setBatchStatus({ running: true, total: 0, completed: 0, failed: 0, results: [] });
        
        try {
            const res = await fetch('/api/llm_ratings/reevaluate_all', { method: 'POST' });
            const data = await res.json();
            
            if (data.count === 0) {
                message.info('没有已评估的股票');
                setBatchStatus(null);
                return;
            }
            
            message.success(`全部重新评估任务已提交，共 ${data.count} 只股票`);
            setBatchStatus({ running: true, total: data.count, completed: 0, failed: 0, results: [] });
            startBatchPolling();
        } catch (error) {
            setBatchStatus(null);
            message.error(`提交失败: ${error.message}`);
        }
    };

    // Normalize: get display action from record (supports both new action and legacy rating fields)
    const getAction = (record) => {
        if (typeof record === 'string') {
            // Called with a raw value (from table column render)
            const val = record.toLowerCase();
            // Map legacy rating to action
            const legacyMap = { good: 'buy_now', bad: 'avoid', neutral: 'wait' };
            return legacyMap[val] || val;
        }
        // Called with a record object
        const action = record?.action?.toLowerCase();
        if (action && ['buy_now', 'wait', 'avoid'].includes(action)) return action;
        // Fallback to legacy rating field
        const rating = record?.rating?.toLowerCase();
        const legacyMap = { good: 'buy_now', bad: 'avoid', neutral: 'wait' };
        return legacyMap[rating] || 'wait';
    };

    const getActionTag = (actionOrRecord) => {
        const action = typeof actionOrRecord === 'string' ? getAction(actionOrRecord) : getAction(actionOrRecord);
        const config = {
            buy_now: { color: 'green', icon: <CheckCircleOutlined />, text: '建议进场' },
            avoid: { color: 'red', icon: <CloseCircleOutlined />, text: '建议回避' },
            wait: { color: 'orange', icon: <MinusCircleOutlined />, text: '等待时机' },
        };
        const c = config[action] || config.wait;
        return <Tag color={c.color} icon={c.icon}>{c.text}</Tag>;
    };

    // Legacy alias
    const getRatingTag = getActionTag;

    const getRiskLevelTag = (record) => {
        const riskLevel = record?.risk_level?.toLowerCase();
        if (riskLevel) {
            const config = {
                low: { color: 'green', text: '低风险' },
                medium: { color: 'orange', text: '中风险' },
                high: { color: 'red', text: '高风险' },
            };
            const c = config[riskLevel] || config.medium;
            return <Tag color={c.color}>{c.text}</Tag>;
        }
        return '-';
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
                            <Text type="secondary">建议进场</Text>
                            <div style={{ marginTop: 8 }}>
                                <Tag color={filterActive('good') ? 'blue' : 'green'} style={{ transition: 'all 0.3s' }}>{good}</Tag>
                            </div>
                        </div>
                    </Col>
                    <Col span={4}>
                        <div style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleFilterClick('bad')}>
                            <Text type="secondary">建议回避</Text>
                            <div style={{ marginTop: 8 }}>
                                <Tag color={filterActive('bad') ? 'blue' : 'red'} style={{ transition: 'all 0.3s' }}>{bad}</Tag>
                            </div>
                        </div>
                    </Col>
                    <Col span={4}>
                        <div style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleFilterClick('neutral')}>
                            <Text type="secondary">等待时机</Text>
                            <div style={{ marginTop: 8 }}>
                                <Tag color={filterActive('neutral') ? 'blue' : 'orange'} style={{ transition: 'all 0.3s' }}>{neutral}</Tag>
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
            render: (date) => date || '-',
        },
        {
            title: '进场建议',
            key: 'action',
            width: 110,
            render: (_, record) => record.action || record.rating ? getActionTag(record) : '-',
        },
        {
            title: '风险等级',
            key: 'risk_level',
            width: 100,
            render: (_, record) => getRiskLevelTag(record),
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
            render: (reason, record) => reason || (!record.rating ? <Text type="secondary">未评估</Text> : '-'),
        },
        {
            title: '操作',
            key: 'operations',
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
                    <div style={{ display: 'flex', gap: 8 }}>
                        <Popconfirm
                            title="全部重新评估"
                            description="将对所有已评估的股票重新执行 LLM 评估，是否继续？"
                            onConfirm={handleReevaluateAll}
                            okText="确认"
                            cancelText="取消"
                            disabled={batchStatus?.running}
                        >
                            <Button 
                                icon={batchStatus?.running ? <LoadingOutlined spin /> : <ReloadOutlined />}
                                loading={batchStatus?.running}
                                disabled={batchStatus?.running}
                                type="primary"
                            >
                                全部重新评估
                            </Button>
                        </Popconfirm>
                        <Popconfirm
                            title="重评失败 & 未评估股票"
                            description="将对所有评估失败 + 信号中未评估的股票重新执行 LLM 评估，是否继续？"
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
                                批量重评失败&未评估股票
                            </Button>
                        </Popconfirm>
                    </div>
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
                            setFilterRating(null);
                            setPage(1);
                            if (isSearchMode && searchedSymbol) {
                                handleStockSelect(searchedSymbol);
                            } else {
                                loadRatings(1, pageSizeRef.current, null, value);
                            }
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
                    <Button 
                        type="primary"
                        icon={<ReloadOutlined />}
                        loading={loading}
                        onClick={handleRefreshList}
                    >
                        查询
                    </Button>
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
                                    当前筛选: {filterRating === 'good' ? '建议进场' : filterRating === 'bad' ? '建议回避' : '等待时机'}
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
                        <StockRatingDetailModal
                            vtSymbol={detailModal.rating?.vt_symbol}
                            open={detailModal.visible}
                            onClose={() => setDetailModal({ visible: false, rating: null })}
                            signals={signals}
                            signalName={selectedSignal}
                            onUpdated={handleRefreshList}
                        />
                    </>
                ) : (
                    <Empty description="请搜索股票代码查看评估详情" />
                )}
            </Spin>
        </div>
    );
};

export default LlmEvaluation;
