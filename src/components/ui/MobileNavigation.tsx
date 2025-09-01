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

  const getNavItems = (userRole: string = '') => {
    // Base items that all roles can see
    const baseItems = [
      { 
        icon: LayoutDashboard, 
        label: "Dashboard", 
        href: "/",
        isActive: (path: string) => path === '/'
      },
      { 
        icon: UserCheck, 
        label: "Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      }
    ];

    // Role-specific items
    let roleItems = [];

    if (userRole === 'admin') {
      // Admin: Full access
      roleItems = [
        { 
          icon: Users, 
          label: "Students", 
          href: "/students",
          isActive: (path: string) => path === '/students'
        },
        { 
          icon: CalendarClock, 
          label: "Sessions", 
          href: "/schedule",
          isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
        },
        { 
          icon: FileText, 
          label: "Reports", 
          href: "/reports",
          isActive: (path: string) => path === '/reports'
        },
        {
          icon: Book, 
          label: "Subjects", 
          href: "/subjects",
          isActive: (path: string) => path === '/subjects'
        }
      ];
    } else if (userRole === 'ROTC admin') {
      // ROTC Admin: Same as Admin but exclude Subjects
      roleItems = [
        { 
          icon: Users, 
          label: "Students", 
          href: "/students",
          isActive: (path: string) => path === '/students'
        },
        { 
          icon: CalendarClock, 
          label: "Sessions", 
          href: "/schedule",
          isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
        },
        { 
          icon: FileText, 
          label: "Reports", 
          href: "/reports",
          isActive: (path: string) => path === '/reports'
        }
      ];
    } else if (userRole === 'Instructor') {
      // Instructor: Keep current access, ensure Subjects is functional
      roleItems = [
        { 
          icon: Users, 
          label: "Students", 
          href: "/students",
          isActive: (path: string) => path === '/students'
        },
        { 
          icon: CalendarClock, 
          label: "Sessions", 
          href: "/schedule",
          isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
        },
        { 
          icon: FileText, 
          label: "Reports", 
          href: "/reports",
          isActive: (path: string) => path === '/reports'
        },
        {
          icon: Book, 
          label: "Subjects", 
          href: "/subjects",
          isActive: (path: string) => path === '/subjects'
        }
      ];
    } else if (userRole === 'SSG officer') {
      // SSG Officer: Keep current access but remove Subjects
      roleItems = [
        { 
          icon: Users, 
          label: "Students", 
          href: "/students",
          isActive: (path: string) => path === '/students'
        },
        { 
          icon: CalendarClock, 
          label: "Sessions", 
          href: "/schedule",
          isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
        },
        { 
          icon: FileText, 
          label: "Reports", 
          href: "/reports",
          isActive: (path: string) => path === '/reports'
        }
      ];
    } else if (userRole === 'ROTC officer') {
      // ROTC Officer: Only Take Attendance, Profile, Logout
      roleItems = [];
    } else {
      // Default user role: Limited access
      roleItems = [
        { 
          icon: CalendarClock, 
          label: "Sessions", 
          href: "/schedule",
          isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
        }
      ];
    }

    return [...baseItems, ...roleItems];
  };

  const navItems = getNavItems(userRole);

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
