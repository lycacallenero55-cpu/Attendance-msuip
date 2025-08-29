import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { 
  LayoutDashboard, 
  UserCheck, 
  Users, 
  FileText, 
  CalendarClock,
  Book,
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";

interface MobileNavigationProps {
  userRole?: string;
}

export const MobileNavigation = ({ userRole = 'user' }: MobileNavigationProps) => {
  const location = useLocation();

  const baseNavItems = [
    { icon: LayoutDashboard, label: "Dashboard", href: "/" },
    { 
      icon: UserCheck, 
      label: "Attendance", 
      href: "/take-attendance",
      isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
    },
    { icon: Users, label: "Students", href: "/students" },
    { 
      icon: CalendarClock, 
      label: "Sessions", 
      href: "/schedule",
      isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
    },
    { icon: FileText, label: "Reports", href: "/reports" },
  ];

  const roleBasedItems = [
    ...(['admin', 'instructor', 'staff'].includes(userRole) ? [{ icon: Book, label: "Subjects", href: "/subjects" }] : []),
  ];

  const navItems = [...baseNavItems, ...roleBasedItems];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-card border-t border-border flex justify-around items-center h-16 z-30 md:hidden">
      {navItems.map((item) => {
        const isActive = item.isActive 
          ? item.isActive(location.pathname) 
          : location.pathname === item.href;
        const Icon = item.icon;
        
        return (
          <Link 
            key={item.href} 
            to={item.href}
            className={cn(
              "flex flex-col items-center justify-center w-full h-full text-xs",
              isActive ? "text-primary" : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Icon className="w-5 h-5 mb-1" />
            <span>{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
};

export default MobileNavigation;
