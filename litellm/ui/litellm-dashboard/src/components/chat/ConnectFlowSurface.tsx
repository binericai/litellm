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
const signedInKey = (flow: string) => `litellm-connect-signed-in:${flow}`;

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

const idsToPost = (connected: ConnectedServers, signedIn: string[]) => {
  const isPick = (id: string) => !connected.authorized.includes(id) || signedIn.includes(id);
  return connected.on.some(isPick) ? connected.on : [];
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
  const [connected, setConnected] = useState<ConnectedServers | null>(null);
  const [signedIn, setSignedIn] = useState(() => (connectFlow ? stored(signedInKey(connectFlow)) : null) ?? []);
  const recordSignIn = useCallback((serverId: string) => setSignedIn((earlier) => [...earlier, serverId]), []);

  useEffect(() => {
    const picks = connectFlow ? stored(picksKey(connectFlow)) : null;
    if (picks) onChange(picks);
  }, [connectFlow, onChange]);

  const rememberPicks = (servers: string[]) => {
    if (connectFlow) remember(picksKey(connectFlow), servers);
    onChange(servers);
  };

  useEffect(() => {
    if (connectFlow) remember(signedInKey(connectFlow), signedIn);
  }, [connectFlow, signedIn]);

  const connectedServerIds = (): string[] | null => {
    if (flow?.state !== "unscoped" || flow.selectable === false) return [];
    return connected && idsToPost(connected, signedIn);
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
          onConnectedServers={setConnected}
          onSignedIn={recordSignIn}
        />
      )}
    </>
  );
};

export default ConnectFlowSurface;
