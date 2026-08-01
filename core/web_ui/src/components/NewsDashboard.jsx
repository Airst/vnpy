import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Card, Select, Spin, Empty, Tag, Typography, Space, Button, message,
    Row, Col, Tooltip, Badge, Statistic, Divider, Alert
} from 'antd';
import {
    NotificationOutlined, ReloadOutlined, ThunderboltOutlined,
    ArrowUpOutlined, ArrowDownOutlined, ClockCircleOutlined,
    RiseOutlined, FallOutlined, SwapOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import StockRatingDetailModal from './StockRatingDetailModal';

const { Text, Paragraph, Title } = Typography;

// 情绪/影响 → 颜色
const sentimentColor = (s) => ({
    positive: 'green',
    negative: 'red',
    neutral: 'default',
}[s] || 'default');

const sentimentLabel = (s) => ({
    positive: '利好',
    negative: '利空',
    neutral: '中性',
}[s] || s);

const timelinessColor = (t) => ({
    high: 'magenta',
    medium: 'orange',
    low: 'default',
}[t] || 'default');

const timelinessLabel = (t) => ({
    high: '高时效',
    medium: '中时效',
    low: '低时效',
}[t] || t);

// 投资方向 → 颜色/标签
const directionColor = (d) => ({
    bullish: 'green',
    bearish: 'red',
    neutral: 'default',
}[d] || 'default');

const directionLabel = (d) => ({
    bullish: '看多',
    bearish: '看空',
    neutral: '中性',
}[d] || d);

const directionIcon = (d) => {
    if (d === 'bullish') return <RiseOutlined />;
    if (d === 'bearish') return <FallOutlined />;
    return <SwapOutlined />;
};

// 确信度 → 颜色（高确信醒目，低确信淡）
const convictionColor = (c) => {
    if (c === null || c === undefined) return '#8c8c8c';
    if (c >= 0.7) return '#cf1322';
    if (c >= 0.4) return '#fa8c16';
    return '#8c8c8c';
};

const pctColor = (p) => {
    if (p === null || p === undefined) return 'default';
    return p > 0 ? 'green' : p < 0 ? 'red' : 'default';
};

const NewsDashboard = () => {
    const [loading, setLoading] = useState(false);
    const [dates, setDates] = useState([]);
    const [selectedDate, setSelectedDate] = useState(null);
    const [items, setItems] = useState([]);
    const [count, setCount] = useState(0);
    const [meta, setMeta] = useState({});
    const [sectors, setSectors] = useState([]);

    const [filterSector, setFilterSector] = useState(null);
    const [filterSentiment, setFilterSentiment] = useState(null);
    const [filterDirection, setFilterDirection] = useState(null);

    const [collectStatus, setCollectStatus] = useState(null);
    const [collecting, setCollecting] = useState(false);
    const pollRef = useRef(null);

    // 股票详情弹窗（点击代表性个股）
    const [stockModal, setStockModal] = useState({ visible: false, vtSymbol: null });

    const loadDates = useCallback(async () => {
        try {
            const res = await fetch('/api/news/dates');
            const data = await res.json();
            setDates(data.dates || []);
            if (!selectedDate && data.dates && data.dates.length) {
                setSelectedDate(data.dates[0]);
            }
        } catch (e) {
            console.error('load dates failed', e);
        }
    }, [selectedDate]);

    const loadNews = useCallback(async (date) => {
        if (!date) return;
        setLoading(true);
        try {
            const params = new URLSearchParams({ limit: 100 });
            if (filterSector) params.append('sector', filterSector);
            if (filterSentiment) params.append('sentiment', filterSentiment);
            if (filterDirection) params.append('direction', filterDirection);
            const res = await fetch(`/api/news?date=${encodeURIComponent(date)}&${params}`);
            const data = await res.json();
            setItems(data.items || []);
            setCount(data.count || 0);
            setMeta(data.meta || {});
        } catch (e) {
            console.error('load news failed', e);
            message.error('加载资讯失败');
        } finally {
            setLoading(false);
        }
    }, [filterSector, filterSentiment, filterDirection]);

    const loadSectors = useCallback(async (date) => {
        if (!date) return;
        try {
            const res = await fetch(`/api/news/sectors?date=${encodeURIComponent(date)}`);
            const data = await res.json();
            setSectors(data.sectors || []);
        } catch (e) {
            console.error('load sectors failed', e);
        }
    }, []);

    // 初始加载
    useEffect(() => { loadDates(); }, []);
    useEffect(() => {
        if (selectedDate) {
            loadNews(selectedDate);
            loadSectors(selectedDate);
        }
    }, [selectedDate]);
    // 过滤变化重新加载
    useEffect(() => {
        if (selectedDate) loadNews(selectedDate);
    }, [filterSector, filterSentiment, filterDirection]);

    // 轮询采集状态
    const pollStatus = useCallback(async () => {
        try {
            const res = await fetch('/api/news/status');
            const data = await res.json();
            setCollectStatus(data);
            if (data.running) {
                setCollecting(true);
                pollRef.current = setTimeout(pollStatus, 3000);
            } else {
                setCollecting(false);
                if (pollRef.current) { clearTimeout(pollRef.current); pollRef.current = null; }
                // 采集结束刷新
                if (collecting) {
                    loadDates();
                }
            }
        } catch (e) {
            console.error('poll status failed', e);
        }
    }, [collecting, loadDates]);

    useEffect(() => { pollStatus(); return () => { if (pollRef.current) clearTimeout(pollRef.current); }; }, []);

    const handleCollect = async () => {
        try {
            const res = await fetch('/api/news/collect', { method: 'POST' });
            if (res.status === 409) {
                message.warning('采集任务正在执行中');
                setCollecting(true);
                pollStatus();
                return;
            }
            const data = await res.json();
            message.success(`采集任务已提交（${data.date}）`);
            setCollecting(true);
            pollStatus();
        } catch (e) {
            message.error('触发采集失败');
        }
    };

    // 采集结束后刷新数据
    useEffect(() => {
        if (!collecting && collectStatus && collectStatus.last_run && selectedDate !== collectStatus.last_date) {
            // 切到最新采集日
            loadDates().then(() => {
                if (collectStatus.last_date) setSelectedDate(collectStatus.last_date);
            });
        }
    }, [collecting]);

    const sentimentOptions = [
        { value: 'positive', label: '利好' },
        { value: 'negative', label: '利空' },
        { value: 'neutral', label: '中性' },
    ];
    const directionOptions = [
        { value: 'bullish', label: '看多' },
        { value: 'bearish', label: '看空' },
        { value: 'neutral', label: '中性' },
    ];
    const sectorOptions = sectors.map(s => ({
        value: s.sector,
        label: `${s.sector}${s.concept_pct_change !== null && s.concept_pct_change !== undefined ? ` (${s.concept_pct_change > 0 ? '+' : ''}${s.concept_pct_change.toFixed(2)}%)` : ''}`,
    }));

    // 汇总统计
    const posCount = items.filter(it => it.sentiment === 'positive').length;
    const negCount = items.filter(it => it.sentiment === 'negative').length;
    const neuCount = items.filter(it => it.sentiment === 'neutral').length;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, height: '100%' }}>
            {/* 顶栏：控制 + 状态 */}
            <Card bordered={false} size="small">
                <Row gutter={12} align="middle">
                    <Col>
                        <Text type="secondary" style={{ fontSize: 12 }}>采集日期</Text>
                        <Select
                            style={{ width: 160, marginLeft: 8 }}
                            value={selectedDate}
                            onChange={setSelectedDate}
                            options={dates.map(d => ({ value: d, label: d }))}
                            placeholder="选择日期"
                            onDropdownVisibleChange={(open) => { if (open) loadDates(); }}
                        />
                    </Col>
                    <Col>
                        <Text type="secondary" style={{ fontSize: 12 }}>板块</Text>
                        <Select
                            style={{ width: 220, marginLeft: 8 }}
                            value={filterSector}
                            onChange={(v) => setFilterSector(v || null)}
                            options={sectorOptions}
                            placeholder="全部板块"
                            allowClear
                            showSearch
                        />
                    </Col>
                    <Col>
                        <Text type="secondary" style={{ fontSize: 12 }}>情绪</Text>
                        <Select
                            style={{ width: 120, marginLeft: 8 }}
                            value={filterSentiment}
                            onChange={(v) => setFilterSentiment(v || null)}
                            options={sentimentOptions}
                            placeholder="全部"
                            allowClear
                        />
                    </Col>
                    <Col>
                        <Text type="secondary" style={{ fontSize: 12 }}>方向</Text>
                        <Select
                            style={{ width: 120, marginLeft: 8 }}
                            value={filterDirection}
                            onChange={(v) => setFilterDirection(v || null)}
                            options={directionOptions}
                            placeholder="全部"
                            allowClear
                        />
                    </Col>
                    <Col flex="auto" />
                    <Col>
                        <Button
                            type="primary"
                            icon={collecting ? <LoadingIcon /> : <ThunderboltOutlined />}
                            onClick={handleCollect}
                            loading={collecting}
                        >
                            {collecting ? '采集中...' : '立即采集'}
                        </Button>
                    </Col>
                </Row>
                {collecting && collectStatus && (
                    <Alert
                        style={{ marginTop: 8 }}
                        type="info" showIcon banner
                        message={`正在用 LLM 联网搜集最新板块资讯（${collectStatus.message || ''}）...`}
                    />
                )}
            </Card>

            {/* 汇总 */}
            <Row gutter={12}>
                <Col span={6}><Card bordered={false} size="small"><Statistic title="资讯条数" value={count} prefix={<NotificationOutlined />} /></Card></Col>
                <Col span={6}><Card bordered={false} size="small"><Statistic title="利好" value={posCount} valueStyle={{ color: '#3f8600' }} prefix={<RiseOutlined />} /></Card></Col>
                <Col span={6}><Card bordered={false} size="small"><Statistic title="利空" value={negCount} valueStyle={{ color: '#cf1322' }} prefix={<FallOutlined />} /></Card></Col>
                <Col span={6}><Card bordered={false} size="small"><Statistic title="中性" value={neuCount} prefix={<SwapOutlined />} /></Card></Col>
            </Row>

            {/* 资讯卡片列表 */}
            <Spin spinning={loading} style={{ flex: 1, overflow: 'auto' }}>
                {items.length === 0 && !loading ? (
                    <Card bordered={false}><Empty description="暂无资讯，点击「立即采集」拉取最新板块资讯" /></Card>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {items.map((it, idx) => (
                            <NewsCard
                                key={`${it.sector}-${it.title}-${idx}`}
                                item={it}
                                onStockClick={(vs) => setStockModal({ visible: true, vtSymbol: vs })}
                            />
                        ))}
                    </div>
                )}
            </Spin>

            {/* 股票 LLM 评估详情弹窗 */}
            <StockRatingDetailModal
                vtSymbol={stockModal.vtSymbol}
                open={stockModal.visible}
                onClose={() => setStockModal({ visible: false, vtSymbol: null })}
            />
        </div>
    );
};

const LoadingIcon = () => <ReloadOutlined spin />;

const NewsCard = ({ item, onStockClick }) => {
    const pct = item.concept_pct_change;
    // 旧数据无 direction 时由 sentiment 反推，保证兼容
    const dir = item.direction || ({ positive: 'bullish', negative: 'bearish', neutral: 'neutral' }[item.sentiment] || 'neutral');
    const conv = item.conviction;
    const hasConv = conv !== null && conv !== undefined && !isNaN(conv);
    const hasAnalysis = item.thesis || item.transmission_chain || item.expectation_gap
        || (item.catalysts && item.catalysts.length) || (item.risks && item.risks.length);

    return (
        <Card
            bordered={false}
            size="small"
            title={
                <Space wrap size={[6, 6]}>
                    <Tag color={directionColor(dir)} icon={directionIcon(dir)} style={{ margin: 0 }}>
                        {directionLabel(dir)}
                    </Tag>
                    {hasConv && (
                        <Tooltip title="投资确信度（0-1，越高越笃定）">
                            <Tag color={convictionColor(conv)} style={{ margin: 0 }}>
                                确信 {(conv * 100).toFixed(0)}%
                            </Tag>
                        </Tooltip>
                    )}
                    <Tag color={timelinessColor(item.timeliness)} icon={<ClockCircleOutlined />} style={{ margin: 0 }}>
                        {timelinessLabel(item.timeliness)}
                    </Tag>
                    {item.time_horizon && (
                        <Tag style={{ margin: 0 }}>{item.time_horizon}</Tag>
                    )}
                    <Text strong style={{ fontSize: 15 }}>{item.title}</Text>
                </Space>
            }
            extra={
                <Space size={8}>
                    {item.sector && <Tag color="blue">{item.sector}</Tag>}
                    {pct !== null && pct !== undefined && (
                        <Tooltip title="板块当日涨跌幅（dc_concept）">
                            <Tag color={pctColor(pct)} icon={pct > 0 ? <ArrowUpOutlined /> : pct < 0 ? <ArrowDownOutlined /> : null}>
                                {pct > 0 ? '+' : ''}{pct.toFixed(2)}%
                            </Tag>
                        </Tooltip>
                    )}
                    <Text type="secondary" style={{ fontSize: 12 }}>{item.info_date}</Text>
                </Space>
            }
        >
            <Paragraph style={{ marginBottom: 8 }}>{item.summary}</Paragraph>

            {hasAnalysis && (
                <div style={{
                    marginBottom: 8, padding: '6px 10px',
                    background: '#fafafa', borderLeft: '3px solid #1890ff', borderRadius: 2,
                }}>
                    {item.thesis && (
                        <div style={{ marginBottom: 4 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>投资逻辑：</Text>
                            <Text strong style={{ fontSize: 13 }}>{item.thesis}</Text>
                        </div>
                    )}
                    {item.transmission_chain && (
                        <div style={{ marginBottom: 4 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>传导链条：</Text>
                            <Text style={{ fontSize: 13 }}>{item.transmission_chain}</Text>
                        </div>
                    )}
                    {item.expectation_gap && (
                        <div>
                            <Text type="secondary" style={{ fontSize: 12 }}>预期差：</Text>
                            <Text style={{ fontSize: 13 }}>{item.expectation_gap}</Text>
                        </div>
                    )}
                </div>
            )}

            {item.impact && (
                <div style={{ marginBottom: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>影响分析：</Text>
                    <Text>{item.impact}</Text>
                </div>
            )}
            {item.rotation && (
                <div style={{ marginBottom: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}><SwapOutlined /> 轮动含义：</Text>
                    <Text>{item.rotation}</Text>
                </div>
            )}

            {(item.catalysts && item.catalysts.length > 0) && (
                <div style={{ marginBottom: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>催化剂：</Text>
                    {item.catalysts.map((c, i) => <Tag key={i} color="green" style={{ marginBottom: 2 }}>{c}</Tag>)}
                </div>
            )}
            {(item.risks && item.risks.length > 0) && (
                <div style={{ marginBottom: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>风险/证伪：</Text>
                    {item.risks.map((r, i) => <Tag key={i} color="volcano" style={{ marginBottom: 2 }}>{r}</Tag>)}
                </div>
            )}

            {(item.related_sectors && item.related_sectors.length > 0) && (
                <div style={{ marginBottom: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>关联板块：</Text>
                    {item.related_sectors.map((s, i) => <Tag key={i}>{s}</Tag>)}
                </div>
            )}

            {(item.stock_implications && item.stock_implications.length > 0) && (
                <div style={{ marginBottom: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>影响标的（LLM 推导）：</Text>
                    {item.stock_implications.map((s, i) => (
                        <Tooltip key={i} title={s.logic || null}>
                            <Tag
                                color={s.direction === 'bullish' ? 'red' : 'green'}
                                icon={s.direction === 'bullish' ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                                style={{ marginBottom: 2 }}
                            >
                                {s.name}
                            </Tag>
                        </Tooltip>
                    ))}
                </div>
            )}

            {(item.mapped_stocks && item.mapped_stocks.length > 0) && (
                <div>
                    <Text type="secondary" style={{ fontSize: 12 }}>代表性个股：</Text>
                    {item.mapped_stocks.map((s, i) => (
                        <Tooltip key={i} title="点击查看 LLM 评估详情">
                            <Tag
                                color={s.direction === 'bullish' ? 'red' : s.direction === 'bearish' ? 'green' : 'geekblue'}
                                style={{ marginBottom: 2, cursor: 'pointer' }}
                                onClick={() => onStockClick && onStockClick(s.vt_symbol)}
                            >
                                {s.vt_symbol} {s.name}
                                {s.pct_chg !== null && s.pct_chg !== undefined && ` ${s.pct_chg > 0 ? '+' : ''}${s.pct_chg.toFixed(1)}%`}
                            </Tag>
                        </Tooltip>
                    ))}
                </div>
            )}

            {item.source && (
                <div style={{ marginTop: 6 }}>
                    <Text type="secondary" style={{ fontSize: 11 }}>来源：{item.source}</Text>
                </div>
            )}
        </Card>
    );
};

export default NewsDashboard;
