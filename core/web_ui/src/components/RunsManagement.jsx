import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    Card, Table, Tag, Button, Space, Popconfirm, Descriptions, Tabs,
    message, Typography, Alert, Empty, Spin, DatePicker, InputNumber, Select, Breadcrumb
} from 'antd';
import {
    ReloadOutlined, CheckCircleOutlined, DeleteOutlined,
    PlayCircleOutlined, ExperimentOutlined, RightOutlined,
    BarChartOutlined, LineChartOutlined, ProfileOutlined, SearchOutlined, EyeOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import BacktestResults from './BacktestResults';
import StockDetailTabs from './StockDetailTabs';

const { Text } = Typography;

// ---------------------------------------------------------------
// Tab② 回测分析: 该 run 引用的回测列表 → 点击加载完整 BacktestResults
// ---------------------------------------------------------------
const RunBacktestTab = ({ detail }) => {
    const backtests = detail?.backtests || [];
    const [selected, setSelected] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // 默认加载最新一次回测
    useEffect(() => {
        const existing = backtests.filter(b => b.exists);
        if (existing.length && !selected) {
            loadResult(existing[existing.length - 1].filename);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [detail?.run_id]);

    const loadResult = async (filename) => {
        setSelected(filename);
        setLoading(true);
        try {
            const res = await fetch(`/api/backtest/result/${filename}`);
            if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
            setResult(await res.json());
        } catch (e) {
            message.error(`加载回测失败: ${e.message || e}`);
            setResult(null);
        } finally {
            setLoading(false);
        }
    };

    if (!backtests.length) {
        return <Empty description="该 run 暂无回测记录。设为生产或对 active run 补全后会自动回测并登记。" />;
    }

    // 从文件名提取回测时间戳后缀便于区分
    const btLabel = (f) => {
        const m = f.match(/_(\d{8}_\d{6})\.json$/);
        return m ? dayjs(m[1], 'YYYYMMDD_HHmmss').format('YYYY-MM-DD HH:mm:ss') : f;
    };

    return (
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Card size="small" title={`回测记录 (${backtests.length})`}>
                <Space wrap>
                    {backtests.map(b => (
                        <Button
                            key={b.filename}
                            size="small"
                            type={selected === b.filename ? 'primary' : 'default'}
                            danger={!b.exists}
                            disabled={!b.exists}
                            onClick={() => loadResult(b.filename)}
                        >
                            {btLabel(b.filename)}{!b.exists && ' (文件缺失)'}
                        </Button>
                    ))}
                </Space>
            </Card>
            <Spin spinning={loading}>
                {result ? (
                    <BacktestResults result={result} />
                ) : (
                    !loading && <Empty description="选择上方回测记录查看详情" />
                )}
            </Spin>
        </Space>
    );
};

// ---------------------------------------------------------------
// Tab③ 信号探索: 数据源 = 该 run 的 signal.parquet
//   - 每日 Top-N 排名 (支持任选日期, 自动回退最近信号日)
//   - 个股详情 (复用 StockDetailTabs: K线 + run 信号叠加)
// ---------------------------------------------------------------
const RunSignalTab = ({ detail }) => {
    const runId = detail.run_id;
    const sigEnd = detail.signal_range?.end;
    const sigStart = detail.signal_range?.start;

    // --- Top-N ---
    const [topDate, setTopDate] = useState(sigEnd ? dayjs(sigEnd) : null);
    const [topN, setTopN] = useState(20);
    const [topData, setTopData] = useState(null);
    const [topLoading, setTopLoading] = useState(false);

    // --- 个股详情 (StockDetailTabs) ---
    const [detailSymbol, setDetailSymbol] = useState(null);
    const [detailName, setDetailName] = useState('');
    const [searchOptions, setSearchOptions] = useState([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const searchTimeoutRef = useRef(null);

    const loadTop = useCallback(async (dateStr, n) => {
        setTopLoading(true);
        try {
            const params = new URLSearchParams();
            if (dateStr) params.set('date', dateStr);
            params.set('n', n);
            const res = await fetch(`/api/runs/${runId}/signal/top?${params}`);
            if (!res.ok) throw new Error((await res.json()).detail);
            const data = await res.json();
            setTopData(data);
            // 非交易日自动回退时同步日期控件
            if (data.date && (!dateStr || data.date !== dateStr)) setTopDate(dayjs(data.date));
            return data;
        } catch (e) {
            message.error(`加载 Top-N 失败: ${e.message || e}`);
            return null;
        } finally {
            setTopLoading(false);
        }
    }, [runId]);

    useEffect(() => {
        // 首次加载 Top-N 后默认展示榜首个股详情
        loadTop(sigEnd || null, 20).then(data => {
            if (data?.items?.length) {
                setDetailSymbol(data.items[0].vt_symbol);
                setDetailName(data.items[0].name || '');
            }
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [runId]);

    const handleSearch = (value) => {
        if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
        if (!value) { setSearchOptions([]); return; }
        searchTimeoutRef.current = setTimeout(() => {
            setSearchLoading(true);
            fetch(`/api/symbols/search?keyword=${value}`)
                .then(res => res.json())
                .then(data => { setSearchOptions(data.symbols || []); setSearchLoading(false); })
                .catch(() => setSearchLoading(false));
        }, 400);
    };

    // Top-N 行联动: 查看个股详情
    const showDetail = (record) => {
        setDetailSymbol(record.vt_symbol);
        setDetailName(record.name || '');
    };

    const topColumns = [
        { title: '排名', dataIndex: 'rank', key: 'rank', width: 48 },
        { title: '代码', dataIndex: 'vt_symbol', key: 'vt_symbol', width: 120, render: v => <Text code>{v}</Text> },
        { title: '名称', dataIndex: 'name', key: 'name', width: 90, render: v => v || <Text type="secondary">-</Text> },
        {
            title: '分数', dataIndex: 'score', key: 'score', width: 80,
            render: v => <Text strong>{v?.toFixed(4)}</Text>,
        },
        {
            title: '', key: 'op', width: 110,
            render: (_, r) => (
                <Button size="small" type="link" icon={<EyeOutlined />} onClick={() => showDetail(r)}>
                    查看详情
                </Button>
            ),
        },
    ];

    return (
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Alert
                type="info"
                showIcon
                message={`信号数据源: 该 run 的信号快照 (${sigStart || '?'} ~ ${sigEnd || '?'})，与生产信号相互独立; K线为行情数据`}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '480px 1fr', gap: 16, alignItems: 'start' }}>
                {/* 每日 Top-N */}
                <Card
                    size="small"
                    title={<Space><ProfileOutlined />每日 Top-N 排名{topData?.date && <Tag color="blue">{topData.date}</Tag>}</Space>}
                >
                    <Space style={{ marginBottom: 12 }}>
                        <DatePicker
                            value={topDate}
                            onChange={(d) => { setTopDate(d); if (d) loadTop(d.format('YYYY-MM-DD'), topN); }}
                            disabledDate={d => (sigStart && d.isBefore(dayjs(sigStart))) || (sigEnd && d.isAfter(dayjs(sigEnd)))}
                            allowClear={false}
                        />
                        <InputNumber min={5} max={100} value={topN} onChange={setTopN} addonBefore="Top" />
                        <Button
                            icon={<SearchOutlined />}
                            onClick={() => loadTop(topDate?.format('YYYY-MM-DD'), topN)}
                            loading={topLoading}
                        >查询</Button>
                    </Space>
                    <Table
                        rowKey="vt_symbol"
                        size="small"
                        columns={topColumns}
                        dataSource={topData?.items || []}
                        loading={topLoading}
                        pagination={false}
                        scroll={{ y: 480 }}
                        rowClassName={(r) => (r.vt_symbol === detailSymbol ? 'ant-table-row-selected' : '')}
                    />
                </Card>

                {/* 个股详情: K线 + run 信号叠加 (复用 StockDetailTabs) */}
                <Card
                    size="small"
                    title={(
                        <Space>
                            <LineChartOutlined />个股详情
                            {detailSymbol && <Tag color="blue">{detailSymbol}{detailName ? ` ${detailName}` : ''}</Tag>}
                        </Space>
                    )}
                    extra={(
                        <Select
                            showSearch
                            style={{ width: 300 }}
                            placeholder="搜索股票 (代码或名称)"
                            value={detailSymbol}
                            onChange={(v, opt) => {
                                setDetailSymbol(v || null);
                                setDetailName(opt?.label?.split(' ')[1] || '');
                            }}
                            onSearch={handleSearch}
                            filterOption={false}
                            notFoundContent={searchLoading ? <Spin size="small" /> : null}
                            options={searchOptions}
                            allowClear
                        />
                    )}
                >
                    {detailSymbol ? (
                        <StockDetailTabs
                            key={detailSymbol}
                            vtSymbol={detailSymbol}
                            signals={[detail.signal_name]}
                            runId={runId}
                            autoLoad
                        />
                    ) : (
                        <Empty description="在左侧排名点「查看详情」，或搜索股票" style={{ padding: '80px 0' }} />
                    )}
                </Card>
            </div>
        </Space>
    );
};

// ---------------------------------------------------------------
// Tab① 概览: manifest + 窗口模型 + 因子 IC 摘要
// ---------------------------------------------------------------
const RunOverviewTab = ({ detail }) => {
    const factorRows = Object.entries(detail.factors || {})
        .map(([name, m]) => ({ key: name, name, ic: m.ic, icir: m.icir }))
        .sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));

    return (
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Descriptions bordered size="small" column={3}>
                <Descriptions.Item label="版本">{detail.version}</Descriptions.Item>
                <Descriptions.Item label="信号名">{detail.signal_name}</Descriptions.Item>
                <Descriptions.Item label="创建时间">{detail.created_at}</Descriptions.Item>
                <Descriptions.Item label="指数过滤">{detail.config?.index || '全市场'}</Descriptions.Item>
                <Descriptions.Item label="模型后端">{detail.config?.backend}</Descriptions.Item>
                <Descriptions.Item label="重训周期">{detail.config?.retrain_days} 天</Descriptions.Item>
                <Descriptions.Item label="信号覆盖" span={2}>
                    {detail.signal_range?.start
                        ? `${detail.signal_range.start} ~ ${detail.signal_range.end} (${(detail.signal_range.rows || 0).toLocaleString()} 行)`
                        : '无信号'}
                </Descriptions.Item>
                <Descriptions.Item label="因子库 (全局)">
                    {detail.factor_store?.exists
                        ? `factors/${detail.version}.parquet (${detail.factor_store.n_columns} 列, ${detail.factor_store.size_mb} MB)`
                        : '未生成'}
                </Descriptions.Item>
            </Descriptions>

            <Card size="small" title={`窗口模型 (${(detail.window_models || []).length})`}>
                {(detail.window_models || []).length ? (
                    <Space wrap size={4}>
                        {detail.window_models.map(ps => <Tag key={ps}>{ps}</Tag>)}
                    </Space>
                ) : <Text type="secondary">无窗口模型</Text>}
            </Card>

            <Card size="small" title={`因子 IC 摘要 (${factorRows.length}, 按 |IC| 排序)`}>
                {factorRows.length ? (
                    <Table
                        rowKey="key"
                        size="small"
                        pagination={{ pageSize: 20 }}
                        dataSource={factorRows}
                        columns={[
                            { title: '因子', dataIndex: 'name', key: 'name' },
                            {
                                title: 'IC', dataIndex: 'ic', key: 'ic', width: 100,
                                sorter: (a, b) => Math.abs(a.ic) - Math.abs(b.ic),
                                render: v => <Text type={Math.abs(v) >= 0.02 ? 'success' : 'secondary'}>{v?.toFixed(4)}</Text>,
                            },
                            {
                                title: 'ICIR', dataIndex: 'icir', key: 'icir', width: 100,
                                sorter: (a, b) => Math.abs(a.icir) - Math.abs(b.icir),
                                render: v => v?.toFixed(3),
                            },
                        ]}
                    />
                ) : <Text type="secondary">无因子 IC 数据 (run 管理上线前迁移的 run 或补全 run 不含因子摘要)</Text>}
            </Card>
        </Space>
    );
};

// ---------------------------------------------------------------
// Run 研究工作台: 头部操作条 + 概览/回测分析/信号探索 三个 Tab
// ---------------------------------------------------------------
const RunWorkspace = ({ runId, taskStatus, onBack, onActivate, onComplete, onDelete }) => {
    const [detail, setDetail] = useState(null);
    const [loading, setLoading] = useState(false);

    const loadDetail = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch(`/api/runs/${runId}`);
            if (!res.ok) throw new Error((await res.json()).detail);
            setDetail(await res.json());
        } catch (e) {
            message.error(`加载详情失败: ${e.message || e}`);
            onBack();
        } finally {
            setLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [runId]);

    useEffect(() => { loadDetail(); }, [loadDetail]);

    if (loading || !detail) {
        return <Card style={{ flex: 1 }}><Spin style={{ display: 'block', margin: '80px auto' }} /></Card>;
    }

    const tabs = [
        {
            key: 'overview',
            label: <Space size={4}><ProfileOutlined />概览</Space>,
            children: <RunOverviewTab detail={detail} />,
        },
        {
            key: 'backtest',
            label: <Space size={4}><BarChartOutlined />回测分析 ({(detail.backtests || []).length})</Space>,
            children: <RunBacktestTab detail={detail} />,
        },
        {
            key: 'signal',
            label: <Space size={4}><LineChartOutlined />信号探索</Space>,
            children: detail.signal_range?.start
                ? <RunSignalTab detail={detail} />
                : <Empty description="该 run 尚无信号，先执行增量补全" />,
        },
    ];

    return (
        <Card
            style={{ flex: 1 }}
            title={
                <Space>
                    <Breadcrumb items={[
                        { title: <a onClick={onBack}><ExperimentOutlined /> 训练轮次</a> },
                        { title: <Text strong>{detail.run_id}</Text> },
                    ]} />
                    {detail.is_active ? <Tag color="green">生产 (active)</Tag> : <Tag>归档</Tag>}
                </Space>
            }
            extra={
                <Space>
                    <Button icon={<ReloadOutlined />} onClick={loadDetail}>刷新</Button>
                    <Popconfirm
                        title={`将 ${detail.run_id} 的信号补全到最新交易日?`}
                        description="窗口模型已存在则纯推理; 跨过窗口边界则训练新窗口模型并存入该 run"
                        onConfirm={() => onComplete(detail.run_id)}
                    >
                        <Button icon={<PlayCircleOutlined />} disabled={taskStatus?.running}>增量补全</Button>
                    </Popconfirm>
                    {!detail.is_active && (
                        <Popconfirm
                            title={`将 ${detail.run_id} 设为生产 run?`}
                            description="该 run 的信号会覆盖生产信号文件 (实盘/策略读取路径)"
                            onConfirm={async () => { await onActivate(detail.run_id); loadDetail(); }}
                        >
                            <Button type="primary" ghost icon={<CheckCircleOutlined />}>设为生产</Button>
                        </Popconfirm>
                    )}
                    {!detail.is_active && (
                        <Popconfirm
                            title={`删除 ${detail.run_id}? 该轮全部产物(模型/信号)将被移除，全局因子库不受影响`}
                            onConfirm={async () => { await onDelete(detail.run_id); onBack(); }}
                        >
                            <Button danger icon={<DeleteOutlined />}>删除</Button>
                        </Popconfirm>
                    )}
                </Space>
            }
        >
            <Tabs items={tabs} defaultActiveKey="overview" destroyInactiveTabPane={false} />
        </Card>
    );
};

// ---------------------------------------------------------------
// 训练轮次(Run)研究工作台入口
//   列表 → 点击 run 进入工作台 (概览 / 回测分析 / 信号探索)
// ---------------------------------------------------------------
const RunsManagement = () => {
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(false);
    const [currentRunId, setCurrentRunId] = useState(null);
    const [taskStatus, setTaskStatus] = useState(null);
    const pollRef = useRef(null);

    const loadRuns = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/runs');
            const data = await res.json();
            setRuns(data.runs || []);
        } catch (e) {
            message.error(`加载训练轮次失败: ${e}`);
        } finally {
            setLoading(false);
        }
    }, []);

    const loadTaskStatus = useCallback(async () => {
        try {
            const res = await fetch('/api/runs/complete/status');
            const data = await res.json();
            setTaskStatus(data);
            return data;
        } catch (e) {
            return null;
        }
    }, []);

    useEffect(() => {
        loadRuns();
        loadTaskStatus();
    }, [loadRuns, loadTaskStatus]);

    // 补全任务运行中每 5s 轮询状态, 结束后刷新列表
    useEffect(() => {
        if (taskStatus?.running && !pollRef.current) {
            pollRef.current = setInterval(async () => {
                const s = await loadTaskStatus();
                if (s && !s.running) {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    loadRuns();
                }
            }, 5000);
        }
        return () => {
            if (pollRef.current) {
                clearInterval(pollRef.current);
                pollRef.current = null;
            }
        };
    }, [taskStatus?.running, loadTaskStatus, loadRuns]);

    const handleActivate = async (runId) => {
        try {
            const res = await fetch(`/api/runs/${runId}/activate`, { method: 'POST' });
            if (!res.ok) throw new Error((await res.json()).detail);
            message.success(`已设为生产 run: ${runId} (信号已同步到生产路径)`);
            loadRuns();
        } catch (e) {
            message.error(`激活失败: ${e.message || e}`);
        }
    };

    const handleComplete = async (runId) => {
        try {
            const res = await fetch(`/api/runs/${runId}/complete`, { method: 'POST' });
            if (!res.ok) throw new Error((await res.json()).detail);
            message.success(`已启动增量补全: ${runId} (后台运行)`);
            loadTaskStatus();
        } catch (e) {
            message.error(`补全启动失败: ${e.message || e}`);
        }
    };

    const handleDelete = async (runId) => {
        try {
            const res = await fetch(`/api/runs/${runId}`, { method: 'DELETE' });
            if (!res.ok) throw new Error((await res.json()).detail);
            message.success(`已删除 run: ${runId}`);
            loadRuns();
        } catch (e) {
            message.error(`删除失败: ${e.message || e}`);
        }
    };

    const columns = [
        {
            title: 'Run ID',
            dataIndex: 'run_id',
            key: 'run_id',
            render: (v, r) => (
                <Space>
                    <a onClick={() => setCurrentRunId(v)}><Text strong style={{ color: 'inherit' }}>{v}</Text></a>
                    {r.is_active && <Tag color="green">生产</Tag>}
                </Space>
            ),
        },
        {
            title: '创建时间',
            dataIndex: 'created_at',
            key: 'created_at',
            width: 160,
        },
        {
            title: '配置',
            key: 'config',
            render: (_, r) => {
                const c = r.config || {};
                return (
                    <Space direction="vertical" size={0}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                            index: {c.index || '全市场'}
                        </Text>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                            {c.backend || 'attention'} / 重训{c.retrain_days ?? '-'}天 / ensemble={c.ensemble ?? '-'}
                        </Text>
                    </Space>
                );
            },
        },
        {
            title: '信号覆盖',
            key: 'signal_range',
            render: (_, r) => {
                const s = r.signal_range || {};
                if (!s.start) return <Tag>无信号</Tag>;
                return (
                    <Space direction="vertical" size={0}>
                        <Text style={{ fontSize: 12 }}>{s.start} ~ {s.end}</Text>
                        <Text type="secondary" style={{ fontSize: 12 }}>{(s.rows || 0).toLocaleString()} 行</Text>
                    </Space>
                );
            },
        },
        {
            title: '产物',
            key: 'artifacts',
            render: (_, r) => (
                <Space size={4}>
                    <Tag color="blue">{r.n_models} 窗口模型</Tag>
                    <Tag color={r.n_factors ? 'purple' : 'default'}>{r.n_factors ? `因子IC(${r.n_factors})` : '无因子IC'}</Tag>
                    <Tag color={(r.backtests || []).length ? 'orange' : 'default'}>{(r.backtests || []).length} 回测</Tag>
                </Space>
            ),
        },
        {
            title: '操作',
            key: 'actions',
            width: 160,
            render: (_, r) => (
                <Button type="primary" ghost size="small" onClick={() => setCurrentRunId(r.run_id)}>
                    进入工作台 <RightOutlined />
                </Button>
            ),
        },
    ];

    const taskAlerts = (
        <>
            {taskStatus?.running && (
                <Alert
                    style={{ marginBottom: 16 }}
                    type="info"
                    showIcon
                    message={`增量补全运行中: ${taskStatus.run_id} (开始于 ${taskStatus.started_at})`}
                    description="后台子进程执行 training.py --run, 完成后自动刷新。日志见 log/run_*.log"
                />
            )}
            {taskStatus && !taskStatus.running && taskStatus.returncode !== null && (
                <Alert
                    style={{ marginBottom: 16 }}
                    type={taskStatus.returncode === 0 ? 'success' : 'error'}
                    showIcon
                    closable
                    message={`上次补全任务 (${taskStatus.run_id}): ${taskStatus.message}`}
                />
            )}
        </>
    );

    // 工作台视图
    if (currentRunId) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                {taskAlerts}
                <RunWorkspace
                    runId={currentRunId}
                    taskStatus={taskStatus}
                    onBack={() => { setCurrentRunId(null); loadRuns(); }}
                    onActivate={handleActivate}
                    onComplete={handleComplete}
                    onDelete={handleDelete}
                />
            </div>
        );
    }

    // 列表视图
    return (
        <Card
            title={<Space><ExperimentOutlined /> 训练轮次管理</Space>}
            extra={
                <Button icon={<ReloadOutlined />} onClick={loadRuns} loading={loading}>刷新</Button>
            }
            style={{ flex: 1 }}
        >
            {taskAlerts}
            {runs.length === 0 && !loading ? (
                <Empty description={
                    <span>
                        暂无训练轮次归档。<br />
                        <Text type="secondary">运行 <Text code>python training.py -v15 -t ...</Text> 全量训练后会自动创建 run</Text>
                    </span>
                } />
            ) : (
                <Table
                    rowKey="run_id"
                    columns={columns}
                    dataSource={runs}
                    loading={loading}
                    pagination={{ pageSize: 10 }}
                    size="middle"
                />
            )}
        </Card>
    );
};

export default RunsManagement;
