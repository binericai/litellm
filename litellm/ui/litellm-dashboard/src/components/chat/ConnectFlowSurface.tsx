"use client";

import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import MCPAppsPanel from "@/components/chat/MCPAppsPanel";
import ConnectFlowBanner from "@/components/chat/ConnectFlowBanner";
import { fetchConnectFlow } from "@/components/networking";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const picksKey = (flow: string) => `litellm-connect-picks:${flow}`;

const storedPicks = (flow: string): string[] | null => {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(picksKey(flow)) ?? "null");
    return Array.isArray(stored) ? stored.filter((name): name is string => typeof name === "string") : null;
  } catch {
    return null;
  }
};

/** Renders the sealed gateway connect flow without trusting URL context. */
const ConnectFlowSurface: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const oauthReturn = searchParams.get("mcpOauthReturn");
  const connectFlow = searchParams.get("connect_flow");

  useEffect(() => {
    if (oauthReturn) {
      const url = new URL(window.location.href);
      url.searchParams.delete("mcpOauthReturn");
      router.replace(url.pathname + url.search);
    }
  }, [oauthReturn, router]);

  const flowQuery = {
    queryKey: ["gateway-connect-flow", connectFlow],
    queryFn: () => fetchConnectFlow(connectFlow!),
    enabled: !!connectFlow,
    retry: false,
  };
  const { data: flow, isError, refetch } = useQuery(flowQuery);
  const [connectedServerIds, setConnectedServerIds] = useState<string[] | null>(null);

  useEffect(() => {
    const stored = connectFlow ? storedPicks(connectFlow) : null;
    if (stored) onChange(stored);
  }, [connectFlow, onChange]);

  const rememberPicks = (servers: string[]) => {
    try {
      if (connectFlow) sessionStorage.setItem(picksKey(connectFlow), JSON.stringify(servers));
    } catch {}
    onChange(servers);
  };

  if (connectFlow === null) {
    return <MCPAppsPanel accessToken={accessToken} selectedServers={selectedServers} onChange={onChange} />;
  }

  return (
    <>
      <ConnectFlowBanner
        flowHandle={connectFlow}
        flow={flow}
        accessToken={accessToken}
        onConnected={refetch}
        failed={isError}
        connectedServerIds={connectedServerIds}
      />
      {flow?.state === "unscoped" && (
        <MCPAppsPanel
          accessToken={accessToken}
          selectedServers={selectedServers}
          onChange={rememberPicks}
          connectMode
          onConnectedServerIds={setConnectedServerIds}
        />
      )}
    </>
  );
};

export default ConnectFlowSurface;
