"use client";

import React, { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import MCPAppsPanel, { type ConnectedServers } from "@/components/chat/MCPAppsPanel";
import ConnectFlowBanner from "@/components/chat/ConnectFlowBanner";
import { fetchConnectFlow } from "@/components/networking";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const picksKey = (flow: string) => `litellm-connect-picks:${flow}`;
const onAtStartKey = (flow: string) => `litellm-connect-on-at-start:${flow}`;

const stored = (key: string): string[] | null => {
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(key) ?? "null");
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : null;
  } catch {
    return null;
  }
};

const remember = (key: string, values: string[]) => {
  try {
    sessionStorage.setItem(key, JSON.stringify(values));
  } catch {}
};

const idsToPost = (connected: ConnectedServers, onAtStart: string[]) =>
  connected.on.some((id) => !onAtStart.includes(id)) ? connected.on : [];

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
  const [connected, setConnected] = useState<ConnectedServers | null>(null);
  const [onAtStart, setOnAtStart] = useState(() => (connectFlow ? stored(onAtStartKey(connectFlow)) : null));

  useEffect(() => {
    const picks = connectFlow ? stored(picksKey(connectFlow)) : null;
    if (picks) onChange(picks);
  }, [connectFlow, onChange]);

  const rememberPicks = (servers: string[]) => {
    if (connectFlow) remember(picksKey(connectFlow), servers);
    onChange(servers);
  };

  const reportConnected = useCallback(
    (servers: ConnectedServers | null) => {
      setConnected(servers);
      if (servers === null || onAtStart !== null || connectFlow === null) return;
      remember(onAtStartKey(connectFlow), servers.authorized);
      setOnAtStart(servers.authorized);
    },
    [connectFlow, onAtStart],
  );

  const connectedServerIds = (): string[] | null => {
    if (flow?.state !== "unscoped" || flow.selectable === false) return [];
    if (connected === null || onAtStart === null) return null;
    return idsToPost(connected, onAtStart);
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
        connectedServerIds={connectedServerIds()}
      />
      {flow?.state === "unscoped" && (
        <MCPAppsPanel
          accessToken={accessToken}
          selectedServers={selectedServers}
          onChange={rememberPicks}
          connectMode
          onConnectedServers={reportConnected}
        />
      )}
    </>
  );
};

export default ConnectFlowSurface;
