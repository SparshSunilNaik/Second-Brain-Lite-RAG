import React from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function QueryPanel({ onQueryResult }) {
    const [question, setQuestion] = React.useState('');
    const [loading, setLoading] = React.useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!question.trim()) {
            return;
        }

        setLoading(true);

        try {
            const response = await fetch(`${API_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question.trim() }),
            });

            const data = await response.json();

            if (response.ok) {
                if (onQueryResult) {
                    onQueryResult({
                        answer: data.answer,
                        sources: data.sources,
                    });
                }
            } else {
                if (onQueryResult) {
                    onQueryResult({
                        answer: `Error: ${data.detail || 'Query failed'}`,
                        sources: [],
                    });
                }
            }
        } catch (error) {
            if (onQueryResult) {
                onQueryResult({
                    answer: `Failed to connect to API: ${error.message}`,
                    sources: [],
                });
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="panel query-panel">
            <h2>Ask a Question</h2>
            <form onSubmit={handleSubmit}>
                <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What are my thoughts on machine learning?"
                    disabled={loading}
                />
                <button type="submit" disabled={loading || !question.trim()}>
                    {loading ? (
                        <>
                            <span className="loading"></span>
                            Searching...
                        </>
                    ) : (
                        'Submit Query'
                    )}
                </button>
            </form>
        </div>
    );
}

export default QueryPanel;
