import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tabs, message, Tag, Space, Descriptions } from 'antd';
import { ReloadOutlined, ApiOutlined, SyncOutlined } from '@ant-design/icons';

const TradeDashboard = () => {
    const [loading, setLoading] = useState(false);
    const [connected, setConnected] = useState(false);
    const [resetting, setResetting] = useState(false);
    
    const [accounts, setAccounts] = useState([]);
    const [positions, setPositions] = useState([]);
    const [orders, setOrders] = useState([]);
    const [trades, setTrades] = useState([]);

    useEffect(() => {
        // Try to connect or check status on mount
        handleConnect(false); 
    }, []);

    const handleConnect = async (silent = false) => {
        try {
            const res = await fetch('/api/trade/connect', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success' || data.status === 'already_connected') {
                setConnected(true);
                if (!silent) message.success(data.message);
                // Wait for data sync if just connected
                setTimeout(() => {
                    fetchData(true);
                }, 1000);
            } else {
                if (!silent) message.error(data.message);
            }
        } catch (error) {
            console.error(error);
            if (!silent) message.error('Failed to connect to gateway');
        }
    };

    const handleReset = async () => {
        setResetting(true);
        try {
            const res = await fetch('/api/trade/reset', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                setConnected(true);
                message.success('Reset and Reconnected successfully');
                setAccounts([]);
                setPositions([]);
                setOrders([]);
                setTrades([]);
                setTimeout(() => {
                    fetchData(true);
                }, 1000);
            } else {
                message.error(data.message || 'Failed to reset');
            }
        } catch (error) {
            console.error(error);
            message.error('Failed to reset connection');
        } finally {
            setResetting(false);
        }
    };

    const fetchData = async (force = false) => {
        if (!connected && force !== true) return;
        setLoading(true);
        try {
            const [accRes, posRes, ordRes, trdRes] = await Promise.all([
                fetch('/api/trade/accounts'),
                fetch('/api/trade/positions'),
                fetch('/api/trade/orders'),
                fetch('/api/trade/trades')
            ]);

            const accData = await accRes.json();
            const posData = await posRes.json();
            const ordData = await ordRes.json();
            const trdData = await trdRes.json();

            setAccounts(accData.accounts || []);
            setPositions(posData.positions || []);
            setOrders(ordData.orders || []);
            setTrades(trdData.trades || []);
            
            message.success('Data refreshed');
        } catch (error) {
            console.error(error);
            message.error('Failed to fetch trade data');
        } finally {
            setLoading(false);
        }
    };

    // Columns Definitions
    const accountColumns = [
        { title: 'Account ID', dataIndex: 'accountid', key: 'accountid' },
        { title: 'Balance', dataIndex: 'balance', key: 'balance', render: (val) => val?.toFixed(2) },
        { title: 'Frozen', dataIndex: 'frozen', key: 'frozen', render: (val) => val?.toFixed(2) },
        { title: 'Available', dataIndex: 'available', key: 'available', render: (val) => val?.toFixed(2) },
        { title: 'Gateway', dataIndex: 'gateway_name', key: 'gateway_name' },
    ];

    const positionColumns = [
        { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
        { title: 'Direction', dataIndex: 'direction', key: 'direction' },
        { title: 'Volume', dataIndex: 'volume', key: 'volume' },
        { title: 'Frozen', dataIndex: 'frozen', key: 'frozen' },
        { title: 'Price', dataIndex: 'price', key: 'price', render: (val) => val?.toFixed(3) },
        { title: 'P&L', dataIndex: 'pnl', key: 'pnl', render: (val) => <span style={{ color: val >= 0 ? 'red' : 'green' }}>{val?.toFixed(2)}</span> },
        { title: 'Exchange', dataIndex: 'exchange', key: 'exchange' },
    ];

    const orderColumns = [
        { title: 'Order ID', dataIndex: 'orderid', key: 'orderid' },
        { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
        { title: 'Direction', dataIndex: 'direction', key: 'direction', render: (text) => (
            <Tag color={text === '多' || text === 'Long' ? 'red' : 'green'}>{text}</Tag>
        )},
        { title: 'Offset', dataIndex: 'offset', key: 'offset' },
        { title: 'Price', dataIndex: 'price', key: 'price' },
        { title: 'Volume', dataIndex: 'volume', key: 'volume' },
        { title: 'Traded', dataIndex: 'traded', key: 'traded' },
        { title: 'Status', dataIndex: 'status', key: 'status', render: (text) => {
            let color = 'default';
            if (text === '全部成交' || text === 'AllTraded') color = 'success';
            if (text === '撤单' || text === 'Cancelled') color = 'warning';
            if (text === '拒单' || text === 'Rejected') color = 'error';
            return <Tag color={color}>{text}</Tag>;
        }},
        { title: 'Time', dataIndex: 'datetime', key: 'datetime' },
    ];

    const tradeColumns = [
        { title: 'Trade ID', dataIndex: 'tradeid', key: 'tradeid' },
        { title: 'Order ID', dataIndex: 'orderid', key: 'orderid' },
        { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
        { title: 'Direction', dataIndex: 'direction', key: 'direction', render: (text) => (
            <Tag color={text === '多' || text === 'Long' ? 'red' : 'green'}>{text}</Tag>
        )},
        { title: 'Offset', dataIndex: 'offset', key: 'offset' },
        { title: 'Price', dataIndex: 'price', key: 'price' },
        { title: 'Volume', dataIndex: 'volume', key: 'volume' },
        { title: 'Time', dataIndex: 'datetime', key: 'datetime' },
    ];

    const items = [
        {
            key: '1',
            label: 'Accounts',
            children: <Table dataSource={accounts} columns={accountColumns} rowKey="accountid" size="small" pagination={false} />
        },
        {
            key: '2',
            label: 'Positions',
            children: <Table dataSource={positions} columns={positionColumns} rowKey="vt_position_id" size="small" />
        },
        {
            key: '3',
            label: 'Orders',
            children: <Table dataSource={orders} columns={orderColumns} rowKey="vt_orderid" size="small" />
        },
        {
            key: '4',
            label: 'Trades',
            children: <Table dataSource={trades} columns={tradeColumns} rowKey="vt_tradeid" size="small" />
        },
    ];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
            <Card title="Trade Dashboard" bordered={false}>
                <Space>
                    <Button 
                        type="primary" 
                        icon={<ApiOutlined />} 
                        onClick={() => handleConnect(false)}
                        disabled={connected}
                    >
                        {connected ? 'Connected' : 'Connect Gateway'}
                    </Button>
                    <Button
                        danger
                        icon={<SyncOutlined />}
                        onClick={handleReset}
                        loading={resetting}
                    >
                        Reset & Reconnect
                    </Button>
                    <Button 
                        icon={<ReloadOutlined />} 
                        onClick={fetchData} 
                        loading={loading}
                        disabled={!connected}
                    >
                        Refresh
                    </Button>
                </Space>
            </Card>

            <Card bordered={false} style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                 <Tabs defaultActiveKey="1" items={items} />
            </Card>
        </div>
    );
};

export default TradeDashboard;
