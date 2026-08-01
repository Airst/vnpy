import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Card, Select, Spin, Empty, Tag, Typography, Space, Button, message,
    Row, Col, Tooltip, Badge, Statistic, Alert, Table, Input, Popconfirm,
    Progress, List,
} from 'antd';
import {
    BulbOutlined, ReloadOutlined, ThunderboltOutlined, RiseOutlined,
    FallOutlined, WarningOutlined, PlusOutlined, DeleteOutlined,
    EyeOutlined, FireOutlined, AlertOutlined, CheckCircleOutlined,
    ClockCircleOutlined, LineChartOutlined,
} from '@ant-design/icons';
import StockRatingDetailModal from './StockRatingDetailModal';

const { Text, Paragraph, Title } = Typography;

// 市场基调 → Alert 类型/文案
const toneMeta = (tone) => ({
    risk_on: { type: 'success', label: '进攻（Risk-On）', icon: <RiseOutlined /> },
    risk_off: { type: 'error', label: '防守（Risk-Off）', icon: <FallOutlined /> },
    neutral: { type: 'info', label: '中性观望', icon: <ClockCircleOutlined /> },
}[tone] || { type: 'info', label: tone || '中性', icon: null });

// 风险类型 → 颜色
const riskTypeColor = (t) => ({
    '利空冲击': 'red',
    '情绪退潮': 'orange',
    '拥挤兑现': 'gold',
}[t] || 'red');

const convictionColor = (c) => {
    if (c >= 0.7) return '#cf1322';
    if (c >= 0.5) return '#fa8c16';
    return '#8c8c8c';
};

// 持仓动向关联强度 → 标签（concept 为弱关联，默认折叠）
const relevanceMeta = (r) => ({
    direct: { label: '直接关联', color: 'geekblue' },
    chain: { label: '产业链', color: 'cyan' },
    concept: { label: '仅概念口径', color: 'default' },
}[r] || { label: '产业链', color: 'cyan' });

// 持仓深度跟踪：情绪/动量 → 标签
const sentimentMeta = (s) => ({
    bullish: { label: '情绪偏多', color: 'red' },
    bearish: { label: '情绪偏空', color: 'green' },
    neutral: { label: '情绪中性', color: 'default' },
}[s] || { label: s || '-', color: 'default' });

const momentumMeta = (m) => ({
    '强势': { color: 'volcano' },
    '震荡': { color: 'default' },
    '走弱': { color: 'blue' },
}[m] || { color: 'default' });

const pctText = (v) => (v === null || v === undefined) ? '-' : `${v > 0 ? '+' : ''}${v.toFixed(2)}%`;
const pctStyle = (v) => ({ color: v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined, fontWeight: 600 });

const AdvisorDashboard = () => {
    const [loading, setLoading] = useState(false);
    const [dates, setDates] = useState([]);
    const [selectedDate, setSelectedDate] = useState(null);
    const [advice, setAdvice] = useState(null);

    const [generating, setGenerating] = useState(false);
    const [genStatus, setGenStatus] = useState(null);
    const pollRef = useRef(null);

    // 持仓池
    const [watchlist, setWatchlist] = useState([]);
    const [addQuery, setAddQuery] = useState('');
    const [addNote, setAddNote] = useState('');
    const [adding, setAdding] = useState(false);

    // 持仓深度跟踪刷新
    const [analyzing, setAnalyzing] = useState(false);
    const holdPollRef = useRef(null);

    // 股票详情弹窗
    const [stockModal, setStockModal] = useState({ visible: false, vtSymbol: null });

    const loadDates = useCallback(async () => {
        try {
            const res = await fetch('/api/advice/dates');
            const data = await res.json();
            setDates(data.dates || []);
            if (!selectedDate && data.dates && data.dates.length) {
                setSelectedDate(data.dates[0]);
            }
        } catch (e) { console.error('load advice dates failed', e); }
    }, [selectedDate]);

    const loadAdvice = useCallback(async (date) => {
        setLoading(true);
        try {
            const res = await fetch(`/api/advice${date ? `?date=${encodeURIComponent(date)}` : ''}`);
            const data = await res.json();
            setAdvice(data && data.status === 'ok' ? data : null);
        } catch (e) {
            console.error('load advice failed', e);
        } finally { setLoading(false); }
    }, []);

    const loadWatchlist = useCallback(async () => {
        try {
            const res = await fetch('/api/watchlist');
            const data = await res.json();
            setWatchlist(data.items || []);
        } catch (e) { console.error('load watchlist failed', e); }
    }, []);

    useEffect(() => { loadDates(); loadWatchlist(); }, []);
    useEffect(() => { if (selectedDate) loadAdvice(selectedDate); }, [selectedDate, loadAdvice]);

    // 生成状态轮询
    const startPolling = useCallback(() => {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
            try {
                const res = await fetch('/api/advice/status');
                const st = await res.json();
                setGenStatus(st);
                if (!st.running) {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    setGenerating(false);
                    if (st.error) {
                        message.error(`建议生成失败: ${st.error}`);
                    } else {
                        message.success(st.message || '建议已生成');
                        await loadDates();
                        if (st.last_date) { setSelectedDate(st.last_date); loadAdvice(st.last_date); }
                    }
                }
            } catch (e) { console.error(e); }
        }, 4000);
    }, [loadDates, loadAdvice]);

    useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

    const handleGenerate = async () => {
        try {
            const res = await fetch('/api/advice/generate', { method: 'POST' });
            if (res.status === 409) { message.warning('生成任务正在执行中'); return; }
            if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
            setGenerating(true);
            message.info('已提交生成任务，LLM 正在蒸馏当日资讯（约 1-3 分钟）...');
            startPolling();
        } catch (e) { message.error(`触发失败: ${e.message}`); }
    };

    const handleAddStock = async () => {
        if (!addQuery.trim()) { message.warning('请输入股票代码或名称'); return; }
        setAdding(true);
        try {
            const res = await fetch('/api/watchlist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: addQuery.trim(), note: addNote.trim() }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || '添加失败');
            if (data.status === 'exists') message.info(data.message);
            else message.success(`已加入股票池: ${data.item.name}（下次生成建议时开始跟踪资讯动向）`);
            setAddQuery(''); setAddNote('');
            loadWatchlist();
        } catch (e) { message.error(e.message); }
        finally { setAdding(false); }
    };

    const handleRemoveStock = async (vtSymbol) => {
        try {
            const res = await fetch(`/api/watchlist/${encodeURIComponent(vtSymbol)}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('移除失败');
            message.success('已移除');
            loadWatchlist();
        } catch (e) { message.error(e.message); }
    };

    // 单独刷新持仓深度跟踪（不重跑全量蒸馏）
    const handleAnalyzeHoldings = async () => {
        try {
            const res = await fetch('/api/watchlist/analyze', { method: 'POST' });
            if (res.status === 409) { message.warning('持仓分析正在执行中'); return; }
            if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
            setAnalyzing(true);
            message.info('持仓分析已提交（逐只联网分析，约 1-2 分钟/只）...');
            if (holdPollRef.current) clearInterval(holdPollRef.current);
            holdPollRef.current = setInterval(async () => {
                try {
                    const st = await (await fetch('/api/watchlist/analyze/status')).json();
                    if (!st.running) {
                        clearInterval(holdPollRef.current);
                        holdPollRef.current = null;
                        setAnalyzing(false);
                        if (st.error) message.error(`持仓分析失败: ${st.error}`);
                        else { message.success(st.message || '持仓分析完成'); loadAdvice(selectedDate); }
                    }
                } catch (e) { console.error(e); }
            }, 4000);
        } catch (e) { message.error(`触发失败: ${e.message}`); }
    };

    useEffect(() => () => { if (holdPollRef.current) clearInterval(holdPollRef.current); }, []);

    const alerts = (advice && advice.watchlist_alerts) || [];
    const holdings = (advice && advice.holding_analysis) || [];
    const holdingByName = {};
    holdings.forEach(h => { holdingByName[h.name] = h; });
    // 弱关联（仅概念口径）默认折叠，旧数据无 relevance 视为强关联不隐藏
    const strongAlerts = alerts.filter(a => a.relevance !== 'concept');
    const weakAlerts = alerts.filter(a => a.relevance === 'concept');
    const [showWeak, setShowWeak] = useState(false);
    const alertsByName = {};
    alerts.forEach(a => { (alertsByName[a.name] = alertsByName[a.name] || []).push(a); });
    const riskCount = alerts.filter(a => a.alert === 'risk').length;
    const posCount = alerts.filter(a => a.alert === 'positive').length;

    // 持仓池表格
    const wlColumns = [
        {
            title: '股票', dataIndex: 'name', key: 'name',
            render: (name, r) => (
                <Space size={4}>
                    <a onClick={() => setStockModal({ visible: true, vtSymbol: r.vt_symbol })}>{name}</a>
                    <Text type="secondary" style={{ fontSize: 11 }}>{r.vt_symbol}</Text>
                </Space>
            ),
        },
        { title: '备注', dataIndex: 'note', key: 'note', render: (t) => t || <Text type="secondary">-</Text> },
        {
            title: '情绪/趋势', key: 'sentiment',
            render: (_, r) => {
                const h = holdingByName[r.name];
                if (!h) return <Text type="secondary" style={{ fontSize: 12 }}>-</Text>;
                const sm = sentimentMeta(h.sentiment);
                return (
                    <Space size={4}>
                        <Tag color={sm.color} style={{ margin: 0 }}>{sm.label}</Tag>
                        <Tag color={momentumMeta(h.momentum).color} style={{ margin: 0 }}>{h.momentum}</Tag>
                        {h.quote && <Text style={{ fontSize: 12, ...pctStyle(h.quote.d5) }}>5日{pctText(h.quote.d5)}</Text>}
                    </Space>
                );
            },
        },
        {
            title: '今日动向', key: 'alerts',
            render: (_, r) => {
                const list = alertsByName[r.name] || [];
                if (!list.length) return <Tag>无动向</Tag>;
                return (
                    <Space size={4} wrap>
                        {list.map((a, i) => {
                            const weak = a.relevance === 'concept';
                            return (
                                <Tooltip key={i} title={`[${relevanceMeta(a.relevance).label}] ${a.rationale}`}>
                                    <Tag color={weak ? 'default' : (a.alert === 'risk' ? 'red' : 'green')}
                                         style={weak ? { opacity: 0.65 } : undefined}
                                         icon={a.alert === 'risk' ? <WarningOutlined /> : <RiseOutlined />}>
                                        {a.alert === 'risk' ? '风险' : '利好'}{weak ? '·弱' : ''}
                                    </Tag>
                                </Tooltip>
                            );
                        })}
                    </Space>
                );
            },
        },
        { title: '加入时间', dataIndex: 'added_at', key: 'added_at', render: (t) => <Text type="secondary" style={{ fontSize: 12 }}>{t}</Text> },
        {
            title: '', key: 'op', width: 50,
            render: (_, r) => (
                <Popconfirm title={`移除 ${r.name}？`} onConfirm={() => handleRemoveStock(r.vt_symbol)}>
                    <Button type="text" danger size="small" icon={<DeleteOutlined />} />
                </Popconfirm>
            ),
        },
    ];

    const tone = advice && advice.market_summary ? toneMeta(advice.market_summary.tone) : null;

    return (
        <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* 顶栏 */}
            <Card bordered={false} size="small">
                <Row gutter={12} align="middle">
                    <Col>
                        <Title level={5} style={{ margin: 0 }}><BulbOutlined /> 每日投资建议</Title>
                    </Col>
                    <Col>
                        <Select
                            style={{ width: 150 }} placeholder="选择日期"
                            value={selectedDate} onChange={setSelectedDate}
                            options={dates.map(d => ({ value: d, label: d }))}
                        />
                    </Col>
                    <Col>
                        <Button icon={<ReloadOutlined />} onClick={() => { loadDates(); loadAdvice(selectedDate); loadWatchlist(); }}>刷新</Button>
                    </Col>
                    <Col flex="auto" />
                    <Col>
                        <Space>
                            {advice && <Text type="secondary" style={{ fontSize: 12 }}>
                                基于 {advice.news_count} 条资讯 · 生成于 {advice.generated_at}
                            </Text>}
                            <Button type="primary" icon={generating ? <ReloadOutlined spin /> : <ThunderboltOutlined />}
                                    onClick={handleGenerate} loading={generating}>
                                {generating ? '生成中...' : '生成今日建议'}
                            </Button>
                        </Space>
                    </Col>
                </Row>
                {generating && genStatus && (
                    <Alert style={{ marginTop: 8 }} type="info" showIcon banner
                           message={`LLM 正在蒸馏资讯并扫描持仓动向（${genStatus.message || ''}）...`} />
                )}
            </Card>

            <Spin spinning={loading}>
                {!advice ? (
                    <Card bordered={false}>
                        <Empty description={
                            <span>暂无当日建议。资讯采集完成后会自动生成，也可点击「生成今日建议」手动触发。</span>
                        } />
                    </Card>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {/* 市场基调 + 汇总数字 */}
                        <Alert
                            type={tone.type} showIcon icon={tone.icon}
                            message={<Space><Text strong>市场基调：{tone.label}</Text></Space>}
                            description={advice.market_summary.comment}
                        />
                        <Row gutter={12}>
                            <Col span={6}><Card bordered={false} size="small"><Statistic title="潜力股" value={(advice.top_stocks || []).length} prefix={<FireOutlined style={{ color: '#cf1322' }} />} /></Card></Col>
                            <Col span={6}><Card bordered={false} size="small"><Statistic title="风险/退潮板块" value={(advice.risk_sectors || []).length} prefix={<WarningOutlined style={{ color: '#fa8c16' }} />} /></Card></Col>
                            <Col span={6}><Card bordered={false} size="small"><Statistic title="持仓风险提示" value={riskCount} valueStyle={{ color: riskCount ? '#cf1322' : undefined }} prefix={<AlertOutlined />} /></Card></Col>
                            <Col span={6}><Card bordered={false} size="small"><Statistic title="持仓利好提示" value={posCount} valueStyle={{ color: posCount ? '#3f8600' : undefined }} prefix={<CheckCircleOutlined />} /></Card></Col>
                        </Row>

                        <Row gutter={12}>
                            {/* 潜力股 */}
                            <Col span={14}>
                                <Card bordered={false} size="small"
                                      title={<Space><FireOutlined style={{ color: '#cf1322' }} /><Text strong>今日最具潜力股票</Text></Space>}>
                                    {(advice.top_stocks || []).length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当日证据不足，宁缺毋滥" /> : (
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                            {advice.top_stocks.map((s, idx) => (
                                                <Card key={idx} size="small" type="inner" hoverable
                                                      title={
                                                          <Space wrap>
                                                              <Badge count={idx + 1} style={{ backgroundColor: idx === 0 ? '#cf1322' : '#fa8c16' }} />
                                                              <a style={{ fontSize: 15, fontWeight: 600 }}
                                                                 onClick={() => s.vt_symbol && setStockModal({ visible: true, vtSymbol: s.vt_symbol })}>
                                                                  {s.name}
                                                              </a>
                                                              {s.vt_symbol && <Text type="secondary" style={{ fontSize: 11 }}>{s.vt_symbol}</Text>}
                                                              {s.sector && <Tag color="blue">{s.sector}</Tag>}
                                                              <Tag>{s.time_horizon}</Tag>
                                                          </Space>
                                                      }
                                                      extra={
                                                          <Tooltip title="确信度（多资讯共振才会 0.7+）">
                                                              <Progress type="circle" percent={Math.round(s.conviction * 100)} size={40}
                                                                        strokeColor={convictionColor(s.conviction)} />
                                                          </Tooltip>
                                                      }>
                                                    <Paragraph style={{ marginBottom: 6 }}>
                                                        <Text type="secondary" style={{ fontSize: 12 }}>评估依据：</Text>{s.rationale}
                                                    </Paragraph>
                                                    {s.entry_risk && (
                                                        <div style={{ marginBottom: 6 }}>
                                                            <Text type="secondary" style={{ fontSize: 12 }}><WarningOutlined /> 买入风险：</Text>
                                                            <Text type="warning" style={{ fontSize: 13 }}>{s.entry_risk}</Text>
                                                        </div>
                                                    )}
                                                    {(s.evidence || []).length > 0 && (
                                                        <div>
                                                            <Text type="secondary" style={{ fontSize: 12 }}><EyeOutlined /> 依据资讯：</Text>
                                                            {s.evidence.map((e, i) => <Tag key={i} style={{ marginBottom: 2, whiteSpace: 'normal' }}>{e}</Tag>)}
                                                        </div>
                                                    )}
                                                </Card>
                                            ))}
                                        </div>
                                    )}
                                </Card>
                            </Col>

                            {/* 风险/退潮板块 */}
                            <Col span={10}>
                                <Card bordered={false} size="small"
                                      title={<Space><WarningOutlined style={{ color: '#fa8c16' }} /><Text strong>风险 / 退潮板块</Text></Space>}>
                                    {(advice.risk_sectors || []).length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无明确风险板块" /> : (
                                        <List
                                            dataSource={advice.risk_sectors}
                                            renderItem={(r) => (
                                                <List.Item style={{ display: 'block' }}>
                                                    <Space wrap style={{ marginBottom: 4 }}>
                                                        <Text strong>{r.sector}</Text>
                                                        <Tag color={riskTypeColor(r.risk_type)}>{r.risk_type}</Tag>
                                                    </Space>
                                                    <Paragraph style={{ marginBottom: 4, fontSize: 13 }}>{r.rationale}</Paragraph>
                                                    {(r.evidence || []).map((e, i) => (
                                                        <Tag key={i} style={{ marginBottom: 2, fontSize: 11, whiteSpace: 'normal' }}>{e}</Tag>
                                                    ))}
                                                </List.Item>
                                            )}
                                        />
                                    )}
                                </Card>
                            </Col>
                        </Row>
                    </div>
                )}
            </Spin>

            {/* 持仓股票池 */}
            <Card bordered={false} size="small"
                  title={<Space><EyeOutlined /><Text strong>持仓股票池</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>提交跟单后，每次生成建议时自动扫描相关资讯，提示风险与利好</Text></Space>}>
                <Space style={{ marginBottom: 12 }} wrap>
                    <Input style={{ width: 200 }} placeholder="股票代码或名称（如 600895 / 张江高科）"
                           value={addQuery} onChange={(e) => setAddQuery(e.target.value)} onPressEnter={handleAddStock} />
                    <Input style={{ width: 220 }} placeholder="备注（可选：持仓成本/仓位）"
                           value={addNote} onChange={(e) => setAddNote(e.target.value)} onPressEnter={handleAddStock} />
                    <Button type="primary" icon={<PlusOutlined />} onClick={handleAddStock} loading={adding}>加入跟单</Button>
                    <Button icon={analyzing ? <ReloadOutlined spin /> : <LineChartOutlined />}
                            onClick={handleAnalyzeHoldings} loading={analyzing}>
                        {analyzing ? '分析中...' : '刷新持仓分析'}
                    </Button>
                </Space>
                <Table rowKey="vt_symbol" size="small" columns={wlColumns} dataSource={watchlist}
                       pagination={false} locale={{ emptyText: '暂无跟单股票，输入代码或名称加入' }} />

                {/* 持仓个股深度跟踪：不依赖当日资讯，行情+情绪+走势研判 */}
                {holdings.length > 0 && (
                    <div style={{ marginTop: 12 }}>
                        <Space size={8}>
                            <Text strong style={{ fontSize: 13 }}>持仓个股跟踪分析</Text>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                                行情快照 + LLM 联网研判，不依赖当日资讯命中
                                {advice.holdings_updated_at ? ` · 更新于 ${advice.holdings_updated_at}` : ''}
                            </Text>
                        </Space>
                        <Row gutter={[12, 12]} style={{ marginTop: 8 }}>
                            {holdings.map((h, i) => {
                                const sm = sentimentMeta(h.sentiment);
                                const q = h.quote;
                                return (
                                    <Col span={12} key={i}>
                                        <Card size="small" type="inner"
                                              title={
                                                  <Space wrap size={4}>
                                                      <a onClick={() => setStockModal({ visible: true, vtSymbol: h.vt_symbol })}>
                                                          <Text strong>{h.name}</Text>
                                                      </a>
                                                      {q && q.industry && <Tag color="blue">{q.industry}</Tag>}
                                                      <Tag color={sm.color}>{sm.label}</Tag>
                                                      <Tag color={momentumMeta(h.momentum).color}>{h.momentum}</Tag>
                                                      {!h.llm_ok && <Tooltip title="LLM 分析未成功，仅展示行情快照与规则判断"><Tag>仅行情</Tag></Tooltip>}
                                                  </Space>
                                              }
                                              extra={h.conviction !== null && h.conviction !== undefined &&
                                                  <Tooltip title="研判依据强度"><Text type="secondary" style={{ fontSize: 12 }}>依据 {(h.conviction * 100).toFixed(0)}%</Text></Tooltip>}>
                                            {q && (
                                                <Space size={12} style={{ marginBottom: 6, fontSize: 12 }} wrap>
                                                    <span>收盘 <Text strong>{q.close}</Text></span>
                                                    <span>当日 <Text style={pctStyle(q.d1)}>{pctText(q.d1)}</Text></span>
                                                    <span>5日 <Text style={pctStyle(q.d5)}>{pctText(q.d5)}</Text></span>
                                                    <span>20日 <Text style={pctStyle(q.d20)}>{pctText(q.d20)}</Text></span>
                                                    <span>量能 <Text strong>{q.vol_ratio}x</Text></span>
                                                </Space>
                                            )}
                                            <Paragraph style={{ marginBottom: 6, fontSize: 13 }}>
                                                <Text type="secondary" style={{ fontSize: 12 }}>走势研判：</Text>{h.view}
                                            </Paragraph>
                                            {(h.drivers || []).length > 0 && (
                                                <div style={{ marginBottom: 4 }}>
                                                    <Text type="secondary" style={{ fontSize: 12 }}>驱动：</Text>
                                                    {h.drivers.map((d, j) => <Tag key={j} color="green" style={{ marginBottom: 2 }}>{d}</Tag>)}
                                                </div>
                                            )}
                                            {(h.risks || []).length > 0 && (
                                                <div>
                                                    <Text type="secondary" style={{ fontSize: 12 }}>风险：</Text>
                                                    {h.risks.map((r, j) => <Tag key={j} color="volcano" style={{ marginBottom: 2 }}>{r}</Tag>)}
                                                </div>
                                            )}
                                        </Card>
                                    </Col>
                                );
                            })}
                        </Row>
                    </div>
                )}

                {/* 持仓动向详情：强关联直接展示，弱关联（仅概念口径）默认折叠 */}
                {alerts.length > 0 && (
                    <div style={{ marginTop: 12 }}>
                        <Text strong style={{ fontSize: 13 }}>今日持仓动向详情</Text>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
                            {strongAlerts.map((a, i) => <AlertCard key={i} a={a} />)}
                        </div>
                        {weakAlerts.length > 0 && (
                            <div style={{ marginTop: 8 }}>
                                <Button type="link" size="small" style={{ padding: 0 }}
                                        onClick={() => setShowWeak(!showWeak)}>
                                    {showWeak ? '收起' : '展开'} {weakAlerts.length} 条弱关联提示（仅概念口径，主营无实质关联，参考价值低）
                                </Button>
                                {showWeak && (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8, opacity: 0.8 }}>
                                        {weakAlerts.map((a, i) => <AlertCard key={i} a={a} />)}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </Card>

            <Text type="secondary" style={{ fontSize: 11, textAlign: 'center' }}>
                以上内容由 LLM 基于当日采集资讯自动生成，仅供研究参考，不构成投资建议。
            </Text>

            <StockRatingDetailModal
                vtSymbol={stockModal.vtSymbol}
                open={stockModal.visible}
                onClose={() => setStockModal({ visible: false, vtSymbol: null })}
            />
        </div>
    );
};

// 持仓动向卡片（含关联强度标签）
const AlertCard = ({ a }) => {
    const rel = relevanceMeta(a.relevance);
    return (
        <Alert type={a.alert === 'risk' ? 'error' : 'success'} showIcon
               message={
                   <Space wrap>
                       <Text strong>{a.name}</Text>
                       <Tag color={a.alert === 'risk' ? 'red' : 'green'}>{a.alert === 'risk' ? '风险提示' : '利好提示'}</Tag>
                       <Tooltip title="关联强度：直接关联=资讯点名/主营实质受益；产业链=上下游实质关联；仅概念口径=主营无实质关联">
                           <Tag color={rel.color}>{rel.label}</Tag>
                       </Tooltip>
                       {a.derived && <Tooltip title="由资讯标的推导自动匹配（非 LLM 汇总判断）"><Tag>自动匹配</Tag></Tooltip>}
                   </Space>
               }
               description={
                   <div>
                       <div style={{ fontSize: 13 }}>{a.rationale}</div>
                       {a.action_hint && <div style={{ marginTop: 4 }}><Text type="secondary" style={{ fontSize: 12 }}>提示：</Text><Text style={{ fontSize: 12 }}>{a.action_hint}</Text></div>}
                       {(a.evidence || []).map((e, j) => <Tag key={j} style={{ marginTop: 4, fontSize: 11, whiteSpace: 'normal' }}>{e}</Tag>)}
                   </div>
               } />
    );
};

export default AdvisorDashboard;
