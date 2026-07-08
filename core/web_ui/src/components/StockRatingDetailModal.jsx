import React, { useState, useEffect, useRef } from 'react';
import { Modal, Button, Tag, Typography, Row, Col, Divider, Alert, message, Empty } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, MinusCircleOutlined, ReloadOutlined, LoadingOutlined } from '@ant-design/icons';
import StockDetailTabs from './StockDetailTabs';

const { Text, Paragraph } = Typography;

/**
 * 股票 LLM 评估详情 Modal（共享组件）。
 *
 * 自包含：按 vt_symbol 拉取评估历史、自管重新评估/任务轮询/刷新。
 * 内容复用 LlmEvaluation 的富版本：StockDetailTabs（K线+信号）+ "LLM 评估" Tab
 * （进场建议/风险/置信度/止损/理由/进场时机/风险事件/分析维度/关键因素）。
 *
 * 被 NewsDashboard（代表性个股点击）、LlmEvaluation（详情）复用。
 */
const StockRatingDetailModal = ({ vtSymbol, open, onClose, signals: signalsProp, signalName, onUpdated }) => {
    const [rating, setRating] = useState(null);
    const [loading, setLoading] = useState(false);
    const [taskStatus, setTaskStatus] = useState(null);
    const [signals, setSignals] = useState(signalsProp || []);
    const pollTimerRef = useRef(null);

    // 信号列表（若调用方未提供则自取）
    useEffect(() => {
        if (signalsProp && signalsProp.length > 0) {
            setSignals(signalsProp);
            return;
        }
        fetch('/api/signals')
            .then(res => res.json())
            .then(data => { if (data.signals) setSignals(data.signals); })
            .catch(() => {});
    }, [signalsProp]);

    // 拉取评估
    const fetchRating = async (symbol) => {
        if (!symbol) return;
        setLoading(true);
        setRating(null);
        try {
            const params = new URLSearchParams({ vt_symbol: symbol });
            if (signalName) params.append('signal_name', signalName);
            const res = await fetch(`/api/llm_ratings?${params}`);
            if (res.ok) {
                const data = await res.json();
                const history = data.history || [];
                if (history.length > 0) {
                    setRating(history[history.length - 1]);
                } else {
                    setRating({ vt_symbol: symbol, not_evaluated: true, reason: '该股票尚未被 LLM 评估' });
                }
            } else {
                setRating({ vt_symbol: symbol, not_evaluated: true, reason: '该股票尚未被 LLM 评估' });
            }
        } catch (e) {
            console.error('Failed to load rating', e);
            setRating({ vt_symbol: symbol, not_evaluated: true, reason: '该股票尚未被 LLM 评估' });
        } finally {
            setLoading(false);
        }
    };

    // 打开 / 切换 vtSymbol 时拉取
    useEffect(() => {
        if (open && vtSymbol) {
            setTaskStatus(null);
            if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }
            fetchRating(vtSymbol);
        }
        if (!open) {
            setRating(null);
            setTaskStatus(null);
            if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open, vtSymbol]);

    // 卸载清理
    useEffect(() => {
        return () => { if (pollTimerRef.current) clearInterval(pollTimerRef.current); };
    }, []);

    const startPolling = (symbol) => {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        pollTimerRef.current = setInterval(async () => {
            try {
                const res = await fetch(`/api/llm_ratings/task_status/${symbol}`);
                const data = await res.json();
                setTaskStatus(data);
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollTimerRef.current);
                    pollTimerRef.current = null;
                    if (data.status === 'completed') {
                        message.success(`${symbol} 评估完成`);
                        await fetchRating(symbol);
                        if (onUpdated) onUpdated();
                    }
                }
            } catch (e) {
                console.error('Failed to poll task status', e);
            }
        }, 3000);
    };

    const handleReevaluate = async (symbol, score) => {
        setTaskStatus({ status: 'running', message: 'LLM 评估中...' });
        try {
            const url = `/api/llm_ratings/reevaluate?vt_symbol=${symbol}&score=${score || 0}`;
            const res = await fetch(url, { method: 'POST' });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || '提交失败');
            }
            message.success(`${symbol} 评估任务已提交，后台执行中...`);
            startPolling(symbol);
        } catch (e) {
            setTaskStatus({ status: 'failed', message: e.message });
            message.error(`提交失败: ${e.message}`);
        }
    };

    const handleRefresh = async (symbol) => {
        setTaskStatus(null);
        await fetchRating(symbol);
        message.success(`${symbol} 数据已刷新`);
    };

    // ---- 展示辅助 ----
    const getAction = (r) => {
        const action = r?.action?.toLowerCase();
        if (action && ['buy_now', 'wait', 'avoid'].includes(action)) return action;
        const legacy = r?.rating?.toLowerCase();
        const m = { good: 'buy_now', bad: 'avoid', neutral: 'wait' };
        return m[legacy] || 'wait';
    };

    const getActionTag = (r) => {
        const action = getAction(r);
        const cfg = {
            buy_now: { color: 'green', icon: <CheckCircleOutlined />, text: '建议进场' },
            avoid: { color: 'red', icon: <CloseCircleOutlined />, text: '建议回避' },
            wait: { color: 'orange', icon: <MinusCircleOutlined />, text: '等待时机' },
        };
        const c = cfg[action] || cfg.wait;
        return <Tag color={c.color} icon={c.icon}>{c.text}</Tag>;
    };

    const getRiskLevelTag = (r) => {
        const rl = r?.risk_level?.toLowerCase();
        if (!rl) return '-';
        const cfg = { low: { color: 'green', text: '低风险' }, medium: { color: 'orange', text: '中风险' }, high: { color: 'red', text: '高风险' } };
        const c = cfg[rl] || cfg.medium;
        return <Tag color={c.color}>{c.text}</Tag>;
    };

    const renderEvaluationTab = () => {
        if (loading) return <Empty description="加载中..." image={Empty.PRESENTED_IMAGE_SIMPLE} />;
        if (!rating) return <Empty description="无评估数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />;
        if (rating.not_evaluated) return <Empty description={rating.reason} image={Empty.PRESENTED_IMAGE_SIMPLE} />;

        const dimensions = rating.analysis_dimensions || {};
        const keyFactors = rating.key_factors || [];
        const positiveFactors = keyFactors.filter(f => f.type === 'positive');
        const negativeFactors = keyFactors.filter(f => f.type === 'negative');

        return (
            <>
                {rating.date && (
                    <div style={{ marginBottom: 12 }}>
                        <Text type="secondary">评估日期：</Text>
                        <Text strong>{rating.date}</Text>
                    </div>
                )}
                <div style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 16]}>
                        <Col span={8}>
                            <Text type="secondary">进场建议：</Text>
                            {getActionTag(rating)}
                        </Col>
                        <Col span={8}>
                            <Text type="secondary">风险等级：</Text>
                            {getRiskLevelTag(rating)}
                        </Col>
                        <Col span={8}>
                            <Text type="secondary">置信度：</Text>
                            <Text strong>{rating.confidence != null ? `${(rating.confidence * 100).toFixed(0)}%` : '-'}</Text>
                        </Col>
                    </Row>
                </div>

                {rating.score !== undefined && rating.score !== null && (
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">模型分数：</Text>
                        <Text code>{rating.score.toFixed?.(4) ?? rating.score}</Text>
                    </div>
                )}

                {rating.stop_loss_price && (
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">止损价：</Text>
                        <Text code>{rating.stop_loss_price}</Text>
                    </div>
                )}

                {rating.reason && (
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">评估理由：</Text>
                        <Paragraph style={{ marginTop: 4 }}>{rating.reason}</Paragraph>
                    </div>
                )}

                {/* 进场时机 */}
                {rating.entry_timing && Object.keys(rating.entry_timing).length > 0 && (
                    <>
                        <Divider orientation="left">进场时机</Divider>
                        <div style={{ marginBottom: 8 }}>
                            {rating.entry_timing.recommendation && (
                                <div style={{ marginBottom: 4 }}>
                                    <Text type="secondary">建议：</Text>
                                    <Text strong>{rating.entry_timing.recommendation}</Text>
                                </div>
                            )}
                            {rating.entry_timing.wait_reason && (
                                <div style={{ marginBottom: 4 }}>
                                    <Text type="secondary">等待原因：</Text>
                                    <Text>{rating.entry_timing.wait_reason}</Text>
                                </div>
                            )}
                            {rating.entry_timing.wait_days > 0 && (
                                <div style={{ marginBottom: 4 }}>
                                    <Text type="secondary">建议等待：</Text>
                                    <Text>{rating.entry_timing.wait_days} 天</Text>
                                </div>
                            )}
                            {rating.entry_timing.upcoming_events?.length > 0 && (
                                <div>
                                    <Text type="secondary">即将到来的事件：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {rating.entry_timing.upcoming_events.map((e, i) => (
                                            <li key={i}><Text>{e}</Text></li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </>
                )}

                {/* 风险事件 */}
                {rating.risk_events?.length > 0 && (
                    <>
                        <Divider orientation="left">风险事件</Divider>
                        {rating.risk_events.map((evt, i) => (
                            <div key={i} style={{ marginBottom: 8, padding: '8px 12px', background: '#fafafa', borderRadius: 4, border: '1px solid #f0f0f0' }}>
                                <div>
                                    <Tag color={evt.severity === 'high' ? 'red' : evt.severity === 'medium' ? 'orange' : 'default'}>
                                        {evt.severity === 'high' ? '高' : evt.severity === 'medium' ? '中' : '低'}
                                    </Tag>
                                    {evt.priced_in && <Tag color="default">已定价</Tag>}
                                    <Text strong>{evt.event}</Text>
                                </div>
                                <div style={{ marginTop: 4 }}>
                                    {evt.date && <Text type="secondary" style={{ marginRight: 12 }}>{evt.date}</Text>}
                                    {evt.source && <Text type="secondary">来源：{evt.source}</Text>}
                                </div>
                            </div>
                        ))}
                    </>
                )}

                {/* 分析维度 */}
                {Object.keys(dimensions).length > 0 && (
                    <>
                        <Divider orientation="left">分析维度</Divider>
                        <Row gutter={[8, 12]}>
                            {dimensions.risk_event && (<Col span={12}><Text type="secondary">事件风险：</Text><div>{dimensions.risk_event}</div></Col>)}
                            {dimensions.earnings_quality && (<Col span={12}><Text type="secondary">盈利质量：</Text><div>{dimensions.earnings_quality}</div></Col>)}
                            {dimensions.entry_timing && (<Col span={12}><Text type="secondary">进场时机：</Text><div>{dimensions.entry_timing}</div></Col>)}
                            {dimensions.technical && (<Col span={12}><Text type="secondary">技术面：</Text><div>{dimensions.technical}</div></Col>)}
                            {dimensions.fundamental && (<Col span={12}><Text type="secondary">基本面：</Text><div>{dimensions.fundamental}</div></Col>)}
                            {dimensions.event && (<Col span={12}><Text type="secondary">事件/催化剂：</Text><div>{dimensions.event}</div></Col>)}
                            {dimensions.sentiment && (<Col span={12}><Text type="secondary">情绪催化：</Text><div>{dimensions.sentiment}</div></Col>)}
                        </Row>
                    </>
                )}

                {/* 关键因素 */}
                {(positiveFactors.length > 0 || negativeFactors.length > 0) && (
                    <>
                        <Divider orientation="left">关键因素</Divider>
                        <Row gutter={16}>
                            {positiveFactors.length > 0 && (
                                <Col span={12}>
                                    <Text type="success">正面因素：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {positiveFactors.map((f, i) => (
                                            <li key={i}>
                                                <Text>{f.content}</Text>
                                                {f.info_date && <Text type="secondary" style={{ fontSize: 12, marginLeft: 4 }}>({f.info_date})</Text>}
                                                {f.timeliness && f.timeliness !== 'priced_in' && (
                                                    <Tag style={{ marginLeft: 4, fontSize: 11 }} color={f.timeliness === 'high' ? 'red' : f.timeliness === 'medium' ? 'orange' : 'default'}>
                                                        {f.timeliness === 'high' ? '高冲击' : f.timeliness === 'medium' ? '中等' : '低冲击'}
                                                    </Tag>
                                                )}
                                                {f.timeliness === 'priced_in' && <Tag style={{ marginLeft: 4, fontSize: 11 }}>已定价</Tag>}
                                            </li>
                                        ))}
                                    </ul>
                                </Col>
                            )}
                            {negativeFactors.length > 0 && (
                                <Col span={12}>
                                    <Text type="danger">负面因素：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {negativeFactors.map((f, i) => (
                                            <li key={i}>
                                                <Text>{f.content}</Text>
                                                {f.info_date && <Text type="secondary" style={{ fontSize: 12, marginLeft: 4 }}>({f.info_date})</Text>}
                                                {f.timeliness && f.timeliness !== 'priced_in' && (
                                                    <Tag style={{ marginLeft: 4, fontSize: 11 }} color={f.timeliness === 'high' ? 'red' : f.timeliness === 'medium' ? 'orange' : 'default'}>
                                                        {f.timeliness === 'high' ? '高冲击' : f.timeliness === 'medium' ? '中等' : '低冲击'}
                                                    </Tag>
                                                )}
                                                {f.timeliness === 'priced_in' && <Tag style={{ marginLeft: 4, fontSize: 11 }}>已定价</Tag>}
                                            </li>
                                        ))}
                                    </ul>
                                </Col>
                            )}
                        </Row>
                    </>
                )}

                {rating.error && (
                    <Alert message="LLM 调用错误" description={rating.error} type="error" showIcon style={{ marginTop: 12 }} />
                )}
            </>
        );
    };

    const reevaluateBtnText = taskStatus?.status === 'running' ? '评估中...'
        : taskStatus?.status === 'completed' ? '重新评估'
        : taskStatus?.status === 'failed' ? '重试'
        : '重新评估';

    return (
        <Modal
            title={`${vtSymbol || ''} - 评估详情`}
            open={open}
            onCancel={onClose}
            width={900}
            destroyOnClose
            footer={[
                <Button
                    key="reevaluate"
                    icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : <ReloadOutlined />}
                    onClick={() => handleReevaluate(vtSymbol, rating?.score)}
                    loading={taskStatus?.status === 'running'}
                    disabled={taskStatus?.status === 'running' || !vtSymbol}
                >
                    {reevaluateBtnText}
                </Button>,
                <Button
                    key="refresh"
                    icon={<ReloadOutlined />}
                    onClick={() => handleRefresh(vtSymbol)}
                    disabled={!vtSymbol}
                >
                    刷新
                </Button>,
                <Button key="close" type="primary" onClick={onClose}>关闭</Button>,
            ]}
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
            {vtSymbol ? (
                <StockDetailTabs
                    vtSymbol={vtSymbol}
                    signals={signals}
                    defaultTab="evaluation"
                    extraTabs={[{
                        key: 'evaluation',
                        label: 'LLM 评估',
                        children: renderEvaluationTab(),
                    }]}
                />
            ) : (
                <Empty description="未指定股票" />
            )}
        </Modal>
    );
};

export default StockRatingDetailModal;
