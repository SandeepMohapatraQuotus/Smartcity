import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  Activity,
  ScanEye,
  Sun,
  Car,
  ScanText,
  CloudFog,
  Radio,
  Users,
  Bell,
  Building2,
  UserSearch,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

type Method = "GET" | "POST" | "DELETE" | "WS" | "LIVE";

const methodStyle: Record<Method, string> = {
  GET: "bg-chart-1/15 text-brand",
  POST: "bg-status-live/15 text-status-live",
  DELETE: "bg-status-alert/15 text-status-alert",
  WS: "bg-brand/15 text-brand animate-pulse",
  LIVE: "bg-brand/15 text-brand animate-pulse",
};

function MethodBadge({ method }: { method: Method }) {
  return (
    <span
      className={cn(
        "ml-auto rounded px-1.5 py-0.5 font-mono text-[10px] font-bold tracking-wide",
        methodStyle[method],
      )}
    >
      {method}
    </span>
  );
}

interface NavItem {
  title: string;
  url: string;
  icon: typeof Activity;
  method: Method;
}

const groups: { label: string; items: NavItem[] }[] = [
  {
    label: "Overview",
    items: [{ title: "Dashboard", url: "/", icon: LayoutDashboard, method: "LIVE" }],
  },
  {
    label: "Analysis",
    items: [{ title: "Full Pipeline", url: "/analyse", icon: Activity, method: "POST" }],
  },
  {
    label: "Classifiers",
    items: [
      { title: "Day / Night", url: "/classify", icon: Sun, method: "POST" },
      { title: "Detect", url: "/detect", icon: Car, method: "POST" },
      { title: "Persons & Registry", url: "/persons", icon: UserSearch, method: "POST" },
    ],
  },
  {
    label: "ANPR",
    items: [{ title: "Read Plates", url: "/anpr", icon: ScanText, method: "POST" }],
  },
  {
    label: "Dehazing",
    items: [{ title: "Dehaze Frame", url: "/dehaze", icon: CloudFog, method: "POST" }],
  },
  {
    label: "Stream",
    items: [{ title: "Live Monitor", url: "/", icon: Radio, method: "WS" }],
  },
  {
    label: "Watchlist",
    items: [{ title: "Watchlist", url: "/watchlist", icon: Users, method: "GET" }],
  },
  {
    label: "Events & Alerts",
    items: [
      { title: "Event Buffer", url: "/events", icon: ScanEye, method: "GET" },
      { title: "Alert Feed", url: "/events", icon: Bell, method: "GET" },
    ],
  },
];

export function AppSidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-1">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-brand/15 text-brand">
            <Building2 className="h-5 w-5" />
          </div>
          <span className="font-mono text-sm font-bold tracking-tight group-data-[collapsible=icon]:hidden">
            Smart City
          </span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        {groups.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={pathname === item.url}
                      tooltip={item.title}
                    >
                      <Link to={item.url} className="flex items-center gap-2">
                        <item.icon className="h-4 w-4 shrink-0" />
                        <span className="group-data-[collapsible=icon]:hidden">
                          {item.title}
                        </span>
                        <span className="group-data-[collapsible=icon]:hidden">
                          <MethodBadge method={item.method} />
                        </span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
    </Sidebar>
  );
}
