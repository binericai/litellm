import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPAppsPanel, { type ConnectedServers } from "./MCPAppsPanel";
import { fetchMCPServers, getMCPOAuthUserCredentialStatus, listMCPTools } from "../networking";
import type { MCPOAuthUserCredentialStatus } from "../networking";
import type { MCPServer } from "../mcp_tools/types";
import { setServerRootPath } from "@/lib/serverRootPath";

vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
  listMCPTools: vi.fn(),
  deleteMCPOAuthUserCredential: vi.fn(),
}));

const oauthReturn = vi.hoisted(() => ({ serverId: null as string | null, status: "idle", onSuccess: () => {} }));
vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  pendingUserMcpOAuthServerId: () => oauthReturn.serverId,
  useUserMcpOAuthFlow: ({ serverId, onSuccess }: { serverId: string; onSuccess: () => void }) => {
    if (serverId !== oauthReturn.serverId) return { startOAuthFlow: vi.fn(), status: "idle" };
    oauthReturn.onSuccess = onSuccess;
    return { startOAuthFlow: vi.fn(), status: oauthReturn.status };
  },
}));

const servers = [
  {
    server_id: "s-ext",
    server_name: "external_logo",
    auth_type: "none",
    mcp_info: { server_name: "external_logo", logo_url: "https://cdn.example.com/ext.png" },
  },
  {
    server_id: "s-local",
    server_name: "local_logo",
    auth_type: "none",
    mcp_info: { server_name: "local_logo", logo_url: "/ui/assets/logos/github.svg" },
  },
  {
    server_id: "s-none",
    server_name: "no_logo",
    auth_type: "none",
  },
] as MCPServer[];

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel accessToken="tok" selectedServers={[]} onChange={vi.fn()} />
    </QueryClientProvider>,
  );

describe("MCPAppsPanel logos", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it("resolves backend logo_url values in the server grid", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    expect(await screen.findByText("external_logo")).toBeInTheDocument();
    expect(screen.getByAltText("external_logo logo")).toHaveAttribute("src", "https://cdn.example.com/ext.png");
    expect(screen.getByAltText("local_logo logo")).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");
  });

  it("renders a colored letter avatar for servers without logo_url", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    expect(await screen.findByText("no_logo")).toBeInTheDocument();
    expect(screen.queryByAltText("no_logo logo")).not.toBeInTheDocument();
    expect(screen.getByText("N")).toBeInTheDocument();
  });

  it("resolves the logo_url in the detail header", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    fireEvent.click(await screen.findByText("local_logo"));

    expect(await screen.findByRole("heading", { name: "local_logo" })).toBeInTheDocument();
    expect(screen.getByAltText("local_logo logo")).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");
  });
});

const connectServers = [
  {
    server_id: "s-m2m",
    server_name: "service_tool",
    auth_type: "oauth2",
    oauth2_flow: "client_credentials",
    connected_app_reachable: true,
  },
  {
    server_id: "s-reach",
    server_name: "reachable_srv",
    auth_type: "none",
    connected_app_reachable: true,
  },
  {
    server_id: "s-unreach",
    server_name: "unreachable_srv",
    auth_type: "none",
    connected_app_reachable: false,
  },
] as MCPServer[];

const renderConnectPanel = (connectMode: boolean, selectedServers: string[] = []) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel accessToken="tok" selectedServers={selectedServers} onChange={vi.fn()} connectMode={connectMode} />
    </QueryClientProvider>,
  );

const idsReported = (onConnectedServers: ReturnType<typeof vi.fn>): string[][] =>
  [...new Set(onConnectedServers.mock.calls.map(([servers]) => JSON.stringify(servers?.on ?? null)))]
    .map((ids) => JSON.parse(ids))
    .filter((ids) => ids !== null);

describe("MCPAppsPanel connected-app reachability (LIT-4861)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    oauthReturn.serverId = null;
    oauthReturn.status = "idle";
    oauthReturn.onSuccess = () => {};
  });

  it("reports every server it shows as on, machine-to-machine ones included", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue([
      ...connectServers,
      { server_id: "s-twin", server_name: "reachable_srv", auth_type: "true_passthrough" },
    ] as MCPServer[]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MCPAppsPanel
          accessToken="tok"
          selectedServers={["reachable_srv", "unreachable_srv", "deleted_srv"]}
          onChange={vi.fn()}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(idsReported(onConnectedServers)).toEqual([["s-m2m", "s-reach"]]));
  });

  it("reports nothing while the list is still loading", async () => {
    vi.mocked(fetchMCPServers).mockReturnValue(new Promise(() => {}));
    const onConnectedServers = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MCPAppsPanel
          accessToken="tok"
          selectedServers={["alpha"]}
          onChange={vi.fn()}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(fetchMCPServers).toHaveBeenCalled());
    expect(idsReported(onConnectedServers)).toEqual([]);
  });

  it("reports nothing and says so when the list fails to load in connect mode", async () => {
    vi.mocked(fetchMCPServers).mockRejectedValue(new Error("list failed"));
    const onConnectedServers = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MCPAppsPanel
          accessToken="tok"
          selectedServers={["alpha"]}
          onChange={vi.fn()}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      </QueryClientProvider>,
    );

    expect(
      await screen.findByText("Could not load your MCP servers. Reload the page to try again."),
    ).toBeInTheDocument();
    expect(idsReported(onConnectedServers)).toEqual([]);
  });

  it("reports an OAuth server only once its saved credential has turned it on", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue([
      {
        server_id: "s-oauth",
        server_name: "oauth_srv",
        auth_type: "oauth2",
        oauth2_flow: "authorization_code",
        connected_app_reachable: true,
      },
      ...connectServers,
    ] as MCPServer[]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    let resolveCredential: (status: MCPOAuthUserCredentialStatus) => void = () => {};
    vi.mocked(getMCPOAuthUserCredentialStatus).mockReturnValue(
      new Promise<MCPOAuthUserCredentialStatus>((resolve) => {
        resolveCredential = resolve;
      }),
    );
    const onConnectedServers = vi.fn();
    const Page = () => {
      const [selected, setSelected] = React.useState<string[]>(["reachable_srv"]);
      return (
        <MCPAppsPanel
          accessToken="tok"
          selectedServers={selected}
          onChange={setSelected}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      );
    };

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <Page />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("oauth_srv")).toBeInTheDocument();
    expect(idsReported(onConnectedServers)).toEqual([]);
    await act(async () => resolveCredential({ server_id: "s-oauth", has_credential: true, is_expired: false }));

    expect(idsReported(onConnectedServers)).toEqual([["s-oauth", "s-m2m", "s-reach"]]);
    expect(onConnectedServers).toHaveBeenLastCalledWith({
      on: ["s-oauth", "s-m2m", "s-reach"],
      authorized: ["s-oauth", "s-m2m"],
    });
  });

  const oauthReturnPage = (
    onConnectedServers: (servers: ConnectedServers | null) => void,
    picked: string[] = ["reachable_srv"],
    onSignedIn?: (serverId: string) => void,
  ) => {
    vi.mocked(fetchMCPServers).mockResolvedValue([
      {
        server_id: "s-oauth",
        server_name: "oauth_srv",
        auth_type: "oauth2",
        oauth2_flow: "authorization_code",
        connected_app_reachable: true,
      },
      ...connectServers,
    ] as MCPServer[]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue({
      server_id: "s-oauth",
      has_credential: false,
      is_expired: false,
    } as MCPOAuthUserCredentialStatus);
    const Page = () => {
      const [selected, setSelected] = React.useState<string[]>(picked);
      return (
        <MCPAppsPanel
          accessToken="tok"
          selectedServers={selected}
          onChange={setSelected}
          connectMode
          onConnectedServers={onConnectedServers}
          onSignedIn={onSignedIn}
        />
      );
    };
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const tree = () => (
      <QueryClientProvider client={client}>
        <Page />
      </QueryClientProvider>
    );
    const { rerender } = render(tree());
    return () => rerender(tree());
  };

  it("holds the report until the login returning from OAuth is saved, then includes its server", async () => {
    oauthReturn.serverId = "s-oauth";
    oauthReturn.status = "exchanging";
    const onConnectedServers = vi.fn();
    oauthReturnPage(onConnectedServers);

    expect(await screen.findByText("oauth_srv")).toBeInTheDocument();
    await waitFor(() => expect(getMCPOAuthUserCredentialStatus).toHaveBeenCalled());
    await act(async () => {});
    expect(screen.queryByText(/^Connect(ing…)?$/)).not.toBeInTheDocument();

    await act(async () => {
      oauthReturn.status = "success";
      oauthReturn.onSuccess();
    });

    expect(idsReported(onConnectedServers)).toEqual([["s-oauth", "s-m2m", "s-reach"]]);
  });

  it("names the server signed in to only once the login returning from OAuth is saved", async () => {
    oauthReturn.serverId = "s-oauth";
    oauthReturn.status = "exchanging";
    const onSignedIn = vi.fn();
    oauthReturnPage(vi.fn(), [], onSignedIn);

    expect(await screen.findByText("oauth_srv")).toBeInTheDocument();
    expect(onSignedIn).not.toHaveBeenCalled();

    await act(async () => {
      oauthReturn.status = "success";
      oauthReturn.onSuccess();
    });

    expect(onSignedIn.mock.calls).toEqual([["s-oauth"]]);
  });

  it("reports the earlier picks when the login returning from OAuth fails", async () => {
    oauthReturn.serverId = "s-oauth";
    oauthReturn.status = "exchanging";
    const onConnectedServers = vi.fn();
    const rerenderPage = oauthReturnPage(onConnectedServers);

    expect(await screen.findByText("oauth_srv")).toBeInTheDocument();
    await waitFor(() => expect(getMCPOAuthUserCredentialStatus).toHaveBeenCalled());
    await act(async () => {});
    oauthReturn.status = "error";
    rerenderPage();

    await waitFor(() => expect(idsReported(onConnectedServers)).toEqual([["s-m2m", "s-reach"]]));
  });

  it("leaves out a picked OAuth server whose card still offers Connect", async () => {
    const onConnectedServers = vi.fn();
    oauthReturnPage(onConnectedServers, ["oauth_srv", "reachable_srv"]);

    expect(await screen.findByText("Connect", { exact: true })).toBeInTheDocument();

    await waitFor(() => expect(idsReported(onConnectedServers)).toEqual([["s-m2m", "s-reach"]]));
  });

  it("leaves a login returning from OAuth to the server's own badge outside connect mode", async () => {
    oauthReturn.serverId = "s-oauth";
    oauthReturn.status = "exchanging";
    vi.mocked(fetchMCPServers).mockResolvedValue([
      { server_id: "s-oauth", server_name: "oauth_srv", auth_type: "oauth2", oauth2_flow: "authorization_code" },
    ] as MCPServer[]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue({
      server_id: "s-oauth",
      has_credential: false,
      is_expired: false,
    } as MCPOAuthUserCredentialStatus);

    renderConnectPanel(false);

    expect(await screen.findByText("Connecting…")).toBeInTheDocument();
  });

  it("reports again once a reload succeeds after the list failed to load", async () => {
    vi.mocked(fetchMCPServers).mockRejectedValueOnce(new Error("list failed")).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    const panel = (token: string) => (
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MCPAppsPanel
          accessToken={token}
          selectedServers={["reachable_srv"]}
          onChange={vi.fn()}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      </QueryClientProvider>
    );
    const { rerender } = render(panel("tok"));
    expect(await screen.findByText(/could not load your mcp servers/i)).toBeInTheDocument();

    rerender(panel("tok-refreshed"));

    await waitFor(() => expect(idsReported(onConnectedServers)).toEqual([["s-m2m", "s-reach"]]));
  });

  it("keeps the plain panel's empty text when the list fails to load outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockRejectedValue(new Error("list failed"));

    renderConnectPanel(false);

    expect(await screen.findByText(/no mcp servers configured/i)).toBeInTheDocument();
  });

  it("reports nothing while a Connect is still loading the server's tools", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    const Page = () => {
      const [selected, setSelected] = React.useState<string[]>([]);
      return (
        <MCPAppsPanel
          accessToken="tok"
          selectedServers={selected}
          onChange={setSelected}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      );
    };
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <Page />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("reachable_srv"));
    await waitFor(() => expect(onConnectedServers).toHaveBeenLastCalledWith({ on: ["s-m2m"], authorized: ["s-m2m"] }));
    let finishConnect: (result: { tools: never[] }) => void = () => {};
    vi.mocked(listMCPTools).mockImplementationOnce(() => new Promise((resolve) => (finishConnect = resolve)));
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    await waitFor(() => expect(onConnectedServers).toHaveBeenLastCalledWith(null));
    await act(async () => finishConnect({ tools: [] }));

    expect(onConnectedServers).toHaveBeenLastCalledWith({ on: ["s-m2m", "s-reach"], authorized: ["s-m2m"] });
  });

  it("requests the connected-app view and hides unreachable servers in connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(true, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("reachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, true);
    expect(screen.queryByText("unreachable_srv")).not.toBeInTheDocument();
    expect(screen.getByText("Connected (1)")).toBeInTheDocument();
    expect(screen.getByText("service_tool")).toBeInTheDocument();
    expect(screen.queryByText("Connect", { exact: true })).not.toBeInTheDocument();
    const toolCountFetchedIds = vi.mocked(listMCPTools).mock.calls.map((call) => call[1]);
    expect(toolCountFetchedIds).toContain("s-reach");
    expect(toolCountFetchedIds).not.toContain("s-unreach");
  });

  it("blocks connecting an unsupported server from the detail view in connect mode", async () => {
    const detailServers = [
      ...connectServers,
      {
        server_id: "s-unsup",
        server_name: "unsupported_srv",
        auth_type: "oauth2_token_exchange",
        connected_app_reachable: true,
      },
    ] as MCPServer[];
    vi.mocked(fetchMCPServers).mockResolvedValue(detailServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(true);

    fireEvent.click(await screen.findByText("unsupported_srv"));
    expect(await screen.findByRole("heading", { name: "unsupported_srv" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
    expect(screen.getByText("Not supported on this connection")).toBeInTheDocument();
  });

  it("keeps the detail-view Connect action outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(false);

    fireEvent.click(await screen.findByText("unreachable_srv"));
    expect(await screen.findByRole("heading", { name: "unreachable_srv" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Connect" })).toBeInTheDocument();
  });

  it("ignores the flag and skips no server outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(false, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("unreachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, false);
    expect(screen.queryByText("Not available to connected apps")).not.toBeInTheDocument();
    expect(screen.getByText("Connected (2)")).toBeInTheDocument();
    const toolCountFetchedIds = vi.mocked(listMCPTools).mock.calls.map((call) => call[1]);
    expect(toolCountFetchedIds).toContain("s-unreach");
  });

  const revocable = (reachable: boolean) =>
    [
      { server_id: "s-reach", server_name: "reachable_srv", auth_type: "none", connected_app_reachable: true },
      { server_id: "s-drop", server_name: "revoked_srv", auth_type: "none", connected_app_reachable: reachable },
    ] as MCPServer[];

  const ConnectPanel = ({
    token,
    onChange,
    client,
  }: {
    token: string;
    onChange: (servers: string[]) => void;
    client: QueryClient;
  }) => (
    <QueryClientProvider client={client}>
      <MCPAppsPanel accessToken={token} selectedServers={[]} onChange={onChange} connectMode />
    </QueryClientProvider>
  );

  const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

  it("drops an open detail view when a refetch removes that server from the reachable set", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValueOnce(revocable(true)).mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={vi.fn()} client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    rerender(<ConnectPanel token="tok-refreshed" onChange={vi.fn()} client={client} />);

    await waitFor(() => expect(screen.queryByRole("heading", { name: "revoked_srv" })).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
    expect(screen.getByText("reachable_srv")).toBeInTheDocument();
  });

  it("does not select a server whose Connect finishes after a refetch removed it", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValueOnce(revocable(true)).mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const onChange = vi.fn();
    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={onChange} client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    let finishConnect: (result: { tools: never[] }) => void = () => {};
    vi.mocked(listMCPTools).mockImplementationOnce(() => new Promise((resolve) => (finishConnect = resolve)));
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    rerender(<ConnectPanel token="tok-refreshed" onChange={onChange} client={client} />);
    await waitFor(() => expect(screen.queryByRole("heading", { name: "revoked_srv" })).not.toBeInTheDocument());

    await act(async () => {
      finishConnect({ tools: [] });
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
    expect(screen.getByText("Connected", { exact: false })).toHaveTextContent("Connected");
  });

  it("does not select a server when Connect resolves in the same tick the refetch drops it", async () => {
    let finishRefetch: (servers: MCPServer[]) => void = () => {};
    vi.mocked(fetchMCPServers)
      .mockResolvedValueOnce(revocable(true))
      .mockImplementationOnce(() => new Promise((resolve) => (finishRefetch = resolve)));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const onChange = vi.fn();
    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={onChange} client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    let finishConnect: (result: { tools: never[] }) => void = () => {};
    vi.mocked(listMCPTools).mockImplementationOnce(() => new Promise((resolve) => (finishConnect = resolve)));
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    rerender(<ConnectPanel token="tok-refreshed" onChange={onChange} client={client} />);

    await act(async () => {
      finishRefetch(revocable(false));
      finishConnect({ tools: [] });
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
  });

  it("does not let a superseded list load overwrite the current reachable set", async () => {
    let finishStaleLoad: (servers: MCPServer[]) => void = () => {};
    vi.mocked(fetchMCPServers)
      .mockImplementationOnce(() => new Promise((resolve) => (finishStaleLoad = resolve)))
      .mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={vi.fn()} client={client} />);
    rerender(<ConnectPanel token="tok-refreshed" onChange={vi.fn()} client={client} />);

    expect(await screen.findByText("reachable_srv")).toBeInTheDocument();

    await act(async () => {
      finishStaleLoad(revocable(true));
    });

    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
  });
});
