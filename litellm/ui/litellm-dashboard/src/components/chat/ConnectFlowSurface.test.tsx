import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ConnectFlowSurface from "./ConnectFlowSurface";
import type { ConnectedServers } from "./MCPAppsPanel";
import { fetchConnectFlow } from "@/components/networking";

const { startOAuthFlow, state, onSuccess, panel } = vi.hoisted(() => ({
  startOAuthFlow: vi.fn(),
  panel: {
    onChange: undefined as ((servers: string[]) => void) | undefined,
    onConnectedServers: undefined as ((servers: ConnectedServers | null) => void) | undefined,
    onSignedIn: undefined as ((serverId: string) => void) | undefined,
  },
  onSuccess: { current: undefined as (() => void) | undefined },
  state: { oauthReturn: null as string | null, connectFlow: null as string | null },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => ({
    get: (key: string) => ({ mcpOauthReturn: state.oauthReturn, connect_flow: state.connectFlow })[key] ?? null,
  }),
}));
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  fetchConnectFlow: vi.fn(),
  getProxyBaseUrl: () => "https://gateway.example.com",
}));
vi.mock("@/components/chat/MCPAppsPanel", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/chat/MCPAppsPanel")>()),
  default: ({
    onChange,
    onConnectedServers,
    onSignedIn,
  }: {
    onChange: (servers: string[]) => void;
    onConnectedServers?: (servers: ConnectedServers | null) => void;
    onSignedIn?: (serverId: string) => void;
  }) => {
    panel.onChange = onChange;
    panel.onConnectedServers = onConnectedServers;
    panel.onSignedIn = onSignedIn;
    return <div data-testid="mcp-apps-panel" />;
  },
}));
vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: ({ onSuccess: success }: { onSuccess: () => void }) => {
    onSuccess.current = success;
    return { startOAuthFlow, status: "idle" };
  },
}));

const flow = (state: "unscoped" | "interactive" | "m2m" | "multi" | "stale", connected: boolean | null = null) => ({
  state,
  client_origin: "https://claude.ai",
  server_id: state === "interactive" || state === "m2m" ? "s-design" : null,
  server_name: state === "interactive" || state === "m2m" ? "design_tool" : null,
  connected,
  servers:
    state === "multi"
      ? [
          { server_id: "s-alpha", server_name: "alpha", connected: false },
          { server_id: "s-beta", server_name: "beta", connected: false },
        ]
      : null,
});

const renderSurface = (onChange = vi.fn()) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <ConnectFlowSurface accessToken="token-123" selectedServers={[]} onChange={onChange} />
    </QueryClientProvider>,
  );

afterEach(() => {
  state.oauthReturn = null;
  state.connectFlow = null;
  onSuccess.current = undefined;
  sessionStorage.clear();
  vi.clearAllMocks();
});

describe("ConnectFlowSurface", () => {
  it.each([
    { result: flow("unscoped"), grid: true, finish: true, cancel: false, oauthStarts: 0 },
    { result: flow("interactive", false), grid: false, finish: false, cancel: true, oauthStarts: 1 },
    { result: flow("interactive", true), grid: false, finish: true, cancel: true, oauthStarts: 0 },
    { result: flow("m2m", true), grid: false, finish: true, cancel: true, oauthStarts: 0 },
    { result: flow("stale"), grid: false, finish: false, cancel: true, oauthStarts: 0 },
  ])(
    "renders $result.state without widening its action surface",
    async ({ result, grid, finish, cancel, oauthStarts }) => {
      state.connectFlow = "flow-handle-123";
      vi.mocked(fetchConnectFlow).mockResolvedValue(result);
      renderSurface();

      await screen.findByRole("button", { name: /finish connecting|cancel|connect/i });
      await waitFor(() => expect(startOAuthFlow).toHaveBeenCalledTimes(oauthStarts));
      expect(screen.queryByTestId("mcp-apps-panel") !== null).toBe(grid);
      expect(screen.queryByRole("button", { name: /finish connecting/i }) !== null).toBe(finish);
      expect(screen.queryByRole("button", { name: "Cancel" }) !== null).toBe(cancel);
    },
  );

  it("keeps the grid and Finish hidden until the gateway accepts a handle", () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockReturnValue(new Promise(() => {}));
    renderSurface();

    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveAttribute("value", "deny");
  });

  it("keeps the grid and Finish hidden when flow validation fails", async () => {
    state.connectFlow = "invalid-handle";
    vi.mocked(fetchConnectFlow).mockRejectedValue(new Error("invalid flow"));
    renderSurface();

    await screen.findByRole("button", { name: "Cancel" });
    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
  });

  it("refetches the sealed flow after the vendor connection completes", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow)
      .mockResolvedValueOnce(flow("interactive", false))
      .mockResolvedValueOnce(flow("interactive", true));
    renderSurface();

    await waitFor(() => expect(startOAuthFlow).toHaveBeenCalledOnce());
    await act(async () => onSuccess.current?.());

    await screen.findByRole("button", { name: /finish connecting/i });
  });

  it("renders the ordinary panel without a flow handle", () => {
    renderSurface();
    expect(fetchConnectFlow).not.toHaveBeenCalled();
    expect(screen.getByTestId("mcp-apps-panel")).toBeInTheDocument();
  });

  it("keeps the servers turned on across the OAuth round trip", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    const { unmount } = renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onChange?.(["catalog"]));
    unmount();

    const onChangeAfterReturn = vi.fn();
    renderSurface(onChangeAfterReturn);

    expect(onChangeAfterReturn).toHaveBeenCalledWith(["catalog"]);
  });

  it.each([flow("interactive", true), flow("m2m", true)])(
    "leaves Finish open on a $state flow, which has no panel to wait for",
    async (result) => {
      state.connectFlow = "flow-handle-123";
      vi.mocked(fetchConnectFlow).mockResolvedValue(result);
      renderSurface();

      expect(await screen.findByRole("button", { name: /finish connecting/i })).toBeEnabled();
    },
  );

  it("keeps the panel but posts no picks when the flow cannot be narrowed", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue({ ...flow("unscoped"), selectable: false });
    renderSurface();

    expect(await screen.findByTestId("mcp-apps-panel")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /finish connecting/i })).toBeEnabled();
    expect(screen.getByText(/https:\/\/claude\.ai will get every server you can reach/)).toBeInTheDocument();
    act(() => panel.onConnectedServers?.({ on: ["catalog-id"], authorized: [] }));
    expect(screen.queryByDisplayValue("catalog-id")).not.toBeInTheDocument();
  });

  it("posts nothing while only the servers that were already on are on", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    renderSurface();
    await screen.findByTestId("mcp-apps-panel");

    act(() => panel.onConnectedServers?.({ on: ["tracker-id", "ledger-id"], authorized: ["tracker-id", "ledger-id"] }));

    expect(screen.getByRole("button", { name: /finish connecting/i })).toBeEnabled();
    expect(screen.queryByDisplayValue("tracker-id")).not.toBeInTheDocument();
  });

  it("posts every server that is on once one more is turned on", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    renderSurface();
    await screen.findByTestId("mcp-apps-panel");

    act(() => panel.onConnectedServers?.({ on: ["tracker-id", "catalog-id"], authorized: ["tracker-id"] }));

    expect(screen.getByDisplayValue("tracker-id")).toHaveAttribute("name", "server");
    expect(screen.getByDisplayValue("catalog-id")).toHaveAttribute("name", "server");
  });

  it("counts a login made during this flow as a pick", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onSignedIn?.("tracker-id"));

    act(() => panel.onConnectedServers?.({ on: ["ledger-id", "tracker-id"], authorized: ["ledger-id", "tracker-id"] }));

    expect(screen.getByDisplayValue("ledger-id")).toHaveAttribute("name", "server");
    expect(screen.getByDisplayValue("tracker-id")).toHaveAttribute("name", "server");
  });

  it("still counts that login after a reload", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    const { unmount } = renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onSignedIn?.("tracker-id"));
    unmount();

    renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onConnectedServers?.({ on: ["ledger-id", "tracker-id"], authorized: ["ledger-id", "tracker-id"] }));

    expect(screen.getByDisplayValue("ledger-id")).toHaveAttribute("name", "server");
    expect(screen.getByDisplayValue("tracker-id")).toHaveAttribute("name", "server");
  });

  it("still counts that login after another login in the same flow", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    const { unmount } = renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onSignedIn?.("tracker-id"));
    unmount();

    renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onSignedIn?.("catalog-id"));
    act(() => panel.onConnectedServers?.({ on: ["ledger-id", "tracker-id"], authorized: ["ledger-id", "tracker-id"] }));

    expect(screen.getByDisplayValue("ledger-id")).toHaveAttribute("name", "server");
    expect(screen.getByDisplayValue("tracker-id")).toHaveAttribute("name", "server");
  });

  it("posts nothing once the login made during this flow is disconnected again", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    act(() => panel.onSignedIn?.("tracker-id"));

    act(() => panel.onConnectedServers?.({ on: ["ledger-id"], authorized: ["ledger-id"] }));

    expect(screen.getByRole("button", { name: /finish connecting/i })).toBeEnabled();
    expect(screen.queryByDisplayValue("ledger-id")).not.toBeInTheDocument();
  });

  it("finishes with the ids of the servers the panel shows as connected", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("unscoped"));
    renderSurface();
    await screen.findByTestId("mcp-apps-panel");
    expect(screen.getByRole("button", { name: /finish connecting/i })).toBeDisabled();

    act(() => panel.onConnectedServers?.({ on: ["catalog-id"], authorized: [] }));

    expect(screen.getByDisplayValue("catalog-id")).toHaveAttribute("name", "server");
    expect(screen.getByRole("button", { name: /finish connecting/i })).toBeEnabled();
  });
});
