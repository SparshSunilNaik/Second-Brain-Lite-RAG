import React from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function IndexPanel({ onIndexComplete }) {
    const [status, setStatus] = React.useState(null);
    const [loading, setLoading] = React.useState(false);

    const handleIndex = async () => {
        setLoading(true);
        setStatus(null);

        try {
            const response = await fetch(`${API_URL}/index`, {
                method: 'POST',
            });

            const data = await response.json();

            if (response.ok) {
                setStatus({
                    type: 'success',
                    message: data.message,
                });
                if (onIndexComplete) {
                    onIndexComplete(data.chunks_indexed);
                }
            } else {
                setStatus({
                    type: 'error',
                    message: data.detail || 'Indexing failed',
                });
            }
        } catch (error) {
            setStatus({
                type: 'error',
                message: `Failed to connect to API: ${error.message}`,
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="panel index-panel">
            <h2>Index Notes</h2>
            <button onClick={handleIndex} disabled={loading}>
                {loading ? (
                    <>
                        <span className="loading"></span>
                        Indexing...
                    </>
                ) : (
                    'Index Notes'
                )}
            </button>

            {status && (
                <div className={`status ${status.type}`}>
                    {status.message}
                </div>
            )}
        </div>
    );
}

export default IndexPanel;
