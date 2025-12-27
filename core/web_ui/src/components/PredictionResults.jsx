import React from 'react';
import { Card, Table, Empty, Tag, Space, Descriptions } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';

const PredictionResults = ({ result }) => {
    // Handle both single prediction and list of predictions
    const predictions = Array.isArray(result?.results) ? result.results : [];

    const columns = [
        { 
            title: 'Symbol', 
            dataIndex: 'vt_symbol', 
            key: 'vt_symbol',
            width: 100
        },
        { 
            title: 'Prediction', 
            dataIndex: 'prediction', 
            key: 'prediction',
            width: 120,
            render: (pred) => {
                if (pred === 'BUY') {
                    return <Tag icon={<ArrowUpOutlined />} color="success">{pred}</Tag>;
                } else if (pred === 'SELL') {
                    return <Tag icon={<ArrowDownOutlined />} color="error">{pred}</Tag>;
                } else {
                    return <Tag color="default">{pred}</Tag>;
                }
            }
        },
        {
            title: 'Confidence',
            dataIndex: 'confidence',
            key: 'confidence',
            width: 120,
            render: (conf) => conf ? `${(conf * 100).toFixed(2)}%` : 'N/A'
        },
        {
            title: 'Reason',
            dataIndex: 'reason',
            key: 'reason',
            render: (reason) => reason || '-'
        }
    ];

    // Add key to each prediction for table
    const dataSource = predictions.map((pred, idx) => ({
        ...pred,
        key: idx
    }));

    if (!result || (Array.isArray(predictions) && predictions.length === 0)) {
        return (
            <Card title="Prediction Results" bordered={false} style={{ height: '100%' }}>
                <Empty description="No prediction results available" />
            </Card>
        );
    }

    return (
        <Card 
            title="Next Trading Day Actions" 
            bordered={false}
            style={{ height: '100%' }}
        >
            <div style={{ marginBottom: 20 }}>
                <div style={{ padding: '12px', background: '#f6f8fb', borderRadius: '4px' }}>
                    <h4>Prediction Summary</h4>
                    <Space split="|" style={{ fontSize: '14px' }}>
                        <span><strong>Total Symbols:</strong> {predictions.length}</span>
                        <span><strong>Buy Signals:</strong> {predictions.filter(p => p.prediction === 'BUY').length}</span>
                        <span><strong>Sell Signals:</strong> {predictions.filter(p => p.prediction === 'SELL').length}</span>
                        <span><strong>Hold/Unknown:</strong> {predictions.filter(p => !['BUY', 'SELL'].includes(p.prediction)).length}</span>
                    </Space>
                </div>
            </div>

            <Table
                columns={columns}
                dataSource={dataSource}
                pagination={{ 
                    pageSize: 15,
                    showTotal: (total) => `Total ${total} symbols`
                }}
                size="middle"
                bordered
                scroll={{ x: 800 }}
            />

            {predictions.length > 0 && (
                <div style={{ marginTop: 20, padding: '12px', background: '#fafafa', borderRadius: '4px', borderLeft: '4px solid #1890ff' }}>
                    <h4 style={{ margin: '0 0 8px 0' }}>Action Items</h4>
                    <ul style={{ margin: 0, paddingLeft: '20px' }}>
                        {predictions.filter(p => p.prediction === 'BUY').length > 0 && (
                            <li><strong>Buy:</strong> {predictions.filter(p => p.prediction === 'BUY').map(p => p.vt_symbol).join(', ')}</li>
                        )}
                        {predictions.filter(p => p.prediction === 'SELL').length > 0 && (
                            <li><strong>Sell:</strong> {predictions.filter(p => p.prediction === 'SELL').map(p => p.vt_symbol).join(', ')}</li>
                        )}
                    </ul>
                </div>
            )}
        </Card>
    );
};

export default PredictionResults;
