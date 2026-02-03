import React from 'react';
import Header from './components/Header';
import IndexPanel from './components/IndexPanel';
import QueryPanel from './components/QueryPanel';
import AnswerPanel from './components/AnswerPanel';

function App() {
    const [answer, setAnswer] = React.useState(null);
    const [sources, setSources] = React.useState([]);

    const handleQueryResult = (result) => {
        setAnswer(result.answer);
        setSources(result.sources);
    };

    const handleIndexComplete = (chunksIndexed) => {
        console.log(`Indexed ${chunksIndexed} chunks`);
    };

    return (
        <div className="app">
            <Header />
            <IndexPanel onIndexComplete={handleIndexComplete} />
            <QueryPanel onQueryResult={handleQueryResult} />
            <AnswerPanel answer={answer} sources={sources} />
        </div>
    );
}

export default App;
