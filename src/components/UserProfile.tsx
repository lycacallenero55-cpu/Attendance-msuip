import { User } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar";

export default function UserProfile() {
  const { user } = useAuth();

  if (!user) return null;

  // Get user initials from email
  const getInitials = (email: string) => {
    return email
      .split("@")[0]
      .split(".")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  return (
    <div className="mt-auto pt-4 border-t border-border">
      <div className="flex items-center justify-center">
        <Avatar>
          <AvatarImage src={user.user_metadata?.avatar_url} />
          <AvatarFallback className="bg-primary/10 text-primary">
            {user.email ? getInitials(user.email) : <User className="h-4 w-4" />}
          </AvatarFallback>
        </Avatar>
      </div>
    </div>
  );
}
