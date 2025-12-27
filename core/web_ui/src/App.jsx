import React, { useState, useEffect, useRef } from 'react';
import { 
    Layout, Menu, Button, Card, DatePicker, message, Select, 
    Typography, Space, Spin, InputNumber
} from 'antd';
import { DashboardOutlined, BarChartOutlined, BgColorsOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import BacktestResults from './components/BacktestResults';
import PredictionResults from './components/PredictionResults';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const App = () => {
    const [strategies, setStrategies] = useState([]);
    const [loadingStrategies, setLoadingStrategies] = useState(false);
    const [factors, setFactors] = useState([]);
    const [activeMenu, setActiveMenu] = useState('backtest');
    
    // Backtest State
    const [btStrategy, setBtStrategy] = useState(null);
    const [btSignal, setBtSignal] = useState(null);
    const [btMaxHoldings, setBtMaxHoldings] = useState(1);
    const [btStart, setBtStart] = useState(null);
    const [btEnd, setBtEnd] = useState(null);
    const [btResult, setBtResult] = useState(null);
    const [btLoading, setBtLoading] = useState(false);

    // Prediction State
    const [predStrategy, setPredStrategy] = useState(null);
    const [predResult, setPredResult] = useState(null);
    const [predLoading, setPredLoading] = useState(false);

    // Ingest State
    const [ingestLoading, setIngestLoading] = useState(false);
    const [ingestLogs, setIngestLogs] = useState([]);
    const logsEndRef = useRef(null);

    useEffect(() => {
        loadStrategies();
        loadFactors();
        loadDataRange();
    }, []);

    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [ingestLogs]);

    const loadDataRange = async () => {
        try {
            const res = await fetch('/api/data_range');
            const data = await res.json();
            if (data.start && data.end) {
                setBtStart(dayjs(data.start, "YYYYMMDD"));
                setBtEnd(dayjs(data.end, "YYYYMMDD"));
            }
        } catch (error) {
            console.error("Failed to load data range", error);
        }
    };

    const loadStrategies = async () => {
        setLoadingStrategies(true);
        try {
            const res = await fetch('/strategies');
            const data = await res.json();
            setStrategies(data.strategies.map(s => ({ value: s, label: s })));
        } catch (error) {
            message.error('Failed to load strategies');
        } finally {
            setLoadingStrategies(false);
        }
    };

    const loadFactors = async () => {
        try {
            const res = await fetch('/factors');
            const data = await res.json();
            setFactors(data.factors.map(f => ({ value: f, label: f })));
        } catch (error) {
            message.error('Failed to load factors');
        }
    };

    const handleIngest = async () => {
        setIngestLoading(true);
        setIngestLogs([]);
        try {
            const response = await fetch('/api/alpha/ingest', { method: 'POST' });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const text = decoder.decode(value);
                const lines = text.split('\n').filter(line => line.trim() !== '');
                setIngestLogs(prev => [...prev, ...lines]);
            }
            message.success('Ingestion triggered successfully');
        } catch (error) {
            message.error('Ingestion failed');
            console.error(error);
        } finally {
            setIngestLoading(false);
        }
    };

    const handleBacktest = async () => {
        if (!btStrategy || !btStart || !btEnd) {
            message.warning('Please fill in all backtest fields');
            return;
        }
        setBtLoading(true);
        setBtResult(null);
        try {
            const payload = {
                strategy_name: btStrategy,
                start_date: btStart.format("YYYYMMDD"),
                end_date: btEnd.format("YYYYMMDD"),
                setting: {
                    signal_name: btSignal,
                    max_holdings: btMaxHoldings
                }
            };
            const res = await fetch('/api/backtest', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            setBtResult(data);
            message.success('Backtest completed');
        } catch (error) {
            message.error('Backtest failed');
            console.error(error);
        } finally {
            setBtLoading(false);
        }
    };

    const handlePrediction = async () => {
        if (!predStrategy) {
            message.warning('Please select strategy');
            return;
        }
        setPredLoading(true);
        setPredResult(null);
        try {
            const payload = {
                strategy_name: predStrategy,
            };
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            setPredResult(data);
            message.success('Prediction completed');
        } catch (error) {
            message.error('Prediction failed');
            console.error(error);
        } finally {
            setPredLoading(false);
        }
    };

    const menuItems = [
        {
            key: 'backtest',
            label: 'Backtest',
            icon: <BarChartOutlined />
        },
        {
            key: 'prediction',
            label: 'Prediction',
            icon: <BgColorsOutlined />
        },
        {
            key: 'system',
            label: 'System Management',
            icon: <DashboardOutlined />
        },
    ];

    // System Management Content
    const renderSystemManagement = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
            {/* 操作区 */}
            <Card title="Operations" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }}>
                    <Space>
                        <Button type="primary" onClick={handleIngest} loading={ingestLoading}>
                            Ingest Factor Data
                        </Button>
                        <Button onClick={loadStrategies} loading={loadingStrategies}>
                            Refresh Strategies
                        </Button>
                    </Space>
                </Space>
            </Card>

            {/* 数据展示区 - 系统日志 */}
            <Card title="System Logs" bordered={false} style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <div style={{ 
                    padding: 10, 
                    background: '#1e1e1e', 
                    color: '#00ff00', 
                    fontFamily: 'monospace', 
                    borderRadius: 4, 
                    maxHeight: '100%', 
                    overflowY: 'auto',
                    flex: 1,
                    minHeight: 300,
                    fontSize: '12px'
                }}>
                    {ingestLogs.length === 0 ? (
                        <Text style={{ color: '#666' }}>No logs yet...</Text>
                    ) : (
                        <>
                            {ingestLogs.map((log, index) => (
                                <div key={index} style={{ marginBottom: '4px' }}>{log}</div>
                            ))}
                            <div ref={logsEndRef} />
                        </>
                    )}
                </div>
            </Card>
        </div>
    );

    // Backtest Content
    const renderBacktest = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
            {/* 操作区 */}
            <Card title="Configuration" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Strategy</label>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="Select Strategy"
                            options={strategies}
                            onChange={setBtStrategy}
                            value={btStrategy}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Signal</label>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="Select Signal (Optional)"
                            options={factors}
                            onChange={setBtSignal}
                            value={btSignal}
                            allowClear
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Max Holdings</label>
                        <InputNumber
                            style={{ width: '100%' }}
                            min={1}
                            max={100}
                            value={btMaxHoldings}
                            onChange={setBtMaxHoldings}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Start Date</label>
                        <DatePicker 
                            style={{ width: '100%' }}
                            value={btStart} 
                            onChange={setBtStart} 
                            format="YYYYMMDD"
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>End Date</label>
                        <DatePicker 
                            style={{ width: '100%' }}
                            value={btEnd} 
                            onChange={setBtEnd} 
                            format="YYYYMMDD"
                        />
                    </div>
                    <Button type="primary" onClick={handleBacktest} loading={btLoading} block size="large">
                        Run Backtest
                    </Button>
                </Space>
            </Card>

            {/* 数据展示区 */}
            <Spin spinning={btLoading} style={{ flex: 1, overflow: 'hidden' }}>
                {btResult ? (
                    <div style={{ height: '100%', overflow: 'auto' }}>
                        <BacktestResults result={btResult} />
                    </div>
                ) : (
                    <Card 
                        title="Results" 
                        bordered={false}
                        style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    >
                        <Text type="secondary">Run backtest to see results</Text>
                    </Card>
                )}
            </Spin>
        </div>
    );

    // Prediction Content
    const renderPrediction = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
            {/* 操作区 */}
            <Card title="Configuration" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Strategy</label>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="Select Strategy"
                            options={strategies}
                            onChange={setPredStrategy}
                            value={predStrategy}
                        />
                    </div>
                    <Button type="primary" onClick={handlePrediction} loading={predLoading} block size="large">
                        Run Prediction
                    </Button>
                </Space>
            </Card>

            {/* 数据展示区 */}
            <Spin spinning={predLoading} style={{ flex: 1, overflow: 'hidden' }}>
                {predResult ? (
                    <div style={{ height: '100%', overflow: 'auto' }}>
                        <PredictionResults result={predResult} />
                    </div>
                ) : (
                    <Card 
                        title="Results" 
                        bordered={false}
                        style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    >
                        <Text type="secondary">Run prediction to see results</Text>
                    </Card>
                )}
            </Spin>
        </div>
    );

    const renderContent = () => {
        switch (activeMenu) {
            case 'system':
                return renderSystemManagement();
            case 'backtest':
                return renderBacktest();
            case 'prediction':
                return renderPrediction();
            default:
                return null;
        }
    };

    return (
        <Layout style={{ minHeight: '100vh' }}>
            <Header style={{ 
                background: '#001529', 
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                padding: '0 24px',
                fontSize: '18px',
                fontWeight: 'bold'
            }}>
                A-Share Analysis System
            </Header>
            <Layout style={{ flex: 1 }}>
                <Sider width={200} style={{ background: '#fff' }} collapsible>
                    <Menu
                        mode="vertical"
                        selectedKeys={[activeMenu]}
                        onClick={(e) => setActiveMenu(e.key)}
                        items={menuItems}
                        style={{ border: 'none' }}
                    />
                </Sider>
                <Content style={{ 
                    padding: '24px', 
                    background: '#f0f2f5', 
                    overflow: 'auto',
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                    {renderContent()}
                </Content>
            </Layout>
        </Layout>
    );
};

export default App;
