import React from 'react';

function AnswerPanel({ answer, sources }) {
    if (!answer) {
        return (
            <div className="panel answer-panel">
                <h2>Answer</h2>
                <div className="empty">
                    Ask a question to see the answer here
                </div>
            </div>
        );
    }

    return (
        <div className="panel answer-panel">
            <h2>Answer</h2>
            <div className="answer">{answer}</div>

            {sources && sources.length > 0 && (
                <div className="sources">
                    <h3>Sources</h3>
                    <ul>
                        {sources.map((source, index) => (
                            <li key={index}>
                                <span className="file">{source.file}</span>
                                <span className="meta">
                                    {source.heading && `- ${source.heading} `}
                                    (chunk {source.chunk_index})
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export default AnswerPanel;
