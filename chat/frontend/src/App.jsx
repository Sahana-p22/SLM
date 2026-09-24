import { useState } from "react";
import Dashboard from "./Dashboard";
import ChatWidget from "./ChatWidget";

export default function App() {
  const [refreshSignal, setRefreshSignal] = useState(0);

  return (
    <>
      <Dashboard refreshSignal={refreshSignal} />
      <ChatWidget onAnswered={() => setRefreshSignal((n) => n + 1)} />
    </>
  );
}
