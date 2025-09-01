import * as React from "react"
import { useEditor, EditorContent } from "@tiptap/react"
import StarterKit from "@tiptap/starter-kit"
import Link from "@tiptap/extension-link"
import Image from "@tiptap/extension-image"
import Placeholder from "@tiptap/extension-placeholder"
import Underline from "@tiptap/extension-underline"
import { TextStyle } from "@tiptap/extension-text-style"
import Color from "@tiptap/extension-color"
import TextAlign from "@tiptap/extension-text-align"
import {
  Bold,
  Italic,
  Strikethrough,
  Underline as UnderlineIcon,
  AlignLeft,
  AlignCenter,
  AlignRight,
  AlignJustify,
  Image as ImageIcon,
  Link2,
  Unlink,
  Palette,
  Type,
  List,
  ListOrdered,
  Quote,
  Undo,
  Redo,
  Code,
} from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Toggle } from "@/components/ui/toggle"
import { toast } from "@/components/ui/use-toast"

const COLORS = [
  { name: "Default", value: "inherit" },
  { name: "Red", value: "#ef4444" },
  { name: "Orange", value: "#f97316" },
  { name: "Yellow", value: "#eab308" },
  { name: "Green", value: "#22c55e" },
  { name: "Blue", value: "#3b82f6" },
  { name: "Purple", value: "#8b5cf6" },
  { name: "Pink", value: "#ec4899" },
  { name: "Gray", value: "#6b7280" },
  { name: "Black", value: "#000000" },
]

const FONT_SIZES = [
  { name: "Small", value: "12px" },
  { name: "Normal", value: "16px" },
  { name: "Large", value: "20px" },
  { name: "Extra Large", value: "24px" },
  { name: "Huge", value: "32px" },
]

interface RichTextEditorProps {
  content?: string
  onChange: (content: string) => void
  placeholder?: string
  className?: string
  editable?: boolean
}

export function RichTextEditor({
  content = "",
  onChange,
  placeholder = "Write something amazing...",
  className,
  editable = true,
}: RichTextEditorProps) {
  const [showLinkInput, setShowLinkInput] = React.useState(false)
  const [linkUrl, setLinkUrl] = React.useState("")
  const [showImageInput, setShowImageInput] = React.useState(false)
  const [imageUrl, setImageUrl] = React.useState("")

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3, 4, 5, 6],
        },
        bulletList: {
          HTMLAttributes: {
            class: "list-disc pl-6",
          },
        },
        orderedList: {
          HTMLAttributes: {
            class: "list-decimal pl-6",
          },
        },
        blockquote: {
          HTMLAttributes: {
            class: "border-l-4 border-gray-300 pl-4 py-1 my-2 text-gray-600 italic",
          },
        },
        codeBlock: {
          HTMLAttributes: {
            class: "bg-gray-100 p-2 rounded font-mono text-sm",
          },
        },
      }),
      Link.configure({
        openOnClick: false,
        HTMLAttributes: {
          class: "text-blue-500 underline hover:text-blue-700",
        },
      }),
      Image,
      Underline,
      TextStyle,
      Color,
      TextAlign.configure({
        types: ["heading", "paragraph", "image"],
      }),
      Placeholder.configure({
        placeholder,
      }),
    ],
    content,
    editable,
    onUpdate: ({ editor }) => {
      onChange(editor.getHTML())
    },
  })

  const setLink = React.useCallback(() => {
    if (!editor) return

    const previousUrl = editor.getAttributes("link").href
    const url = linkUrl || previousUrl

    // Empty or invalid URL, remove the link
    if (!url) {
      editor.chain().focus().extendMarkRange("link").unsetLink().run()
      setShowLinkInput(false)
      setLinkUrl("")
      return
    }

    // Valid URL, set the link
    editor
      .chain()
      .focus()
      .extendMarkRange("link")
      .setLink({ href: url })
      .run()

    setShowLinkInput(false)
    setLinkUrl("")
  }, [editor, linkUrl])

  const addImage = React.useCallback(() => {
    if (!editor || !imageUrl) return

    // Basic URL validation
    try {
      new URL(imageUrl)
      editor.chain().focus().setImage({ src: imageUrl }).run()
      setShowImageInput(false)
      setImageUrl("")
    } catch (e) {
      toast({
        title: "Invalid URL",
        description: "Please enter a valid image URL",
        variant: "destructive",
      })
    }
  }, [editor, imageUrl])

  const setTextColor = (color: string) => {
    editor?.chain().focus().setColor(color).run()
  }

  const setFontSize = (size: string) => {
    editor?.chain().focus().setFontSize(size).run()
  }

  if (!editor) {
    return <div className={cn("min-h-[200px] rounded-md border p-4", className)} />
  }

  return (
    <div className={cn("space-y-2", className)}>
      {editable && (
        <div className="flex flex-wrap gap-1 rounded-t-md border border-b-0 bg-muted/50 p-1">
          {/* Text Formatting */}
          <Toggle
            size="sm"
            pressed={editor.isActive("bold")}
            onPressedChange={() => editor.chain().focus().toggleBold().run()}
            disabled={!editor.can().chain().focus().toggleBold().run()}
          >
            <Bold className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("italic")}
            onPressedChange={() => editor.chain().focus().toggleItalic().run()}
            disabled={!editor.can().chain().focus().toggleItalic().run()}
          >
            <Italic className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("underline")}
            onPressedChange={() => editor.chain().focus().toggleUnderline().run()}
          >
            <UnderlineIcon className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("strike")}
            onPressedChange={() => editor.chain().focus().toggleStrike().run()}
            disabled={!editor.can().chain().focus().toggleStrike().run()}
          >
            <Strikethrough className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("code")}
            onPressedChange={() => editor.chain().focus().toggleCode().run()}
            disabled={!editor.can().chain().focus().toggleCode().run()}
          >
            <Code className="h-4 w-4" />
          </Toggle>

          <div className="mx-1 h-6 w-px bg-border" />

          {/* Headings */}
          <Toggle
            size="sm"
            pressed={editor.isActive("heading", { level: 1 })}
            onPressedChange={() =>
              editor.chain().focus().toggleHeading({ level: 1 }).run()
            }
          >
            <span className="text-xs font-bold">H1</span>
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("heading", { level: 2 })}
            onPressedChange={() =>
              editor.chain().focus().toggleHeading({ level: 2 }).run()
            }
          >
            <span className="text-xs font-bold">H2</span>
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("heading", { level: 3 })}
            onPressedChange={() =>
              editor.chain().focus().toggleHeading({ level: 3 }).run()
            }
          >
            <span className="text-xs font-bold">H3</span>
          </Toggle>

          <div className="mx-1 h-6 w-px bg-border" />

          {/* Lists */}
          <Toggle
            size="sm"
            pressed={editor.isActive("bulletList")}
            onPressedChange={() =>
              editor.chain().focus().toggleBulletList().run()
            }
          >
            <List className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("orderedList")}
            onPressedChange={() =>
              editor.chain().focus().toggleOrderedList().run()
            }
          >
            <ListOrdered className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive("blockquote")}
            onPressedChange={() =>
              editor.chain().focus().toggleBlockquote().run()
            }
          >
            <Quote className="h-4 w-4" />
          </Toggle>

          <div className="mx-1 h-6 w-px bg-border" />

          {/* Text Alignment */}
          <Toggle
            size="sm"
            pressed={editor.isActive({ textAlign: "left" })}
            onPressedChange={() =>
              editor.chain().focus().setTextAlign("left").run()
            }
          >
            <AlignLeft className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive({ textAlign: "center" })}
            onPressedChange={() =>
              editor.chain().focus().setTextAlign("center").run()
            }
          >
            <AlignCenter className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive({ textAlign: "right" })}
            onPressedChange={() =>
              editor.chain().focus().setTextAlign("right").run()
            }
          >
            <AlignRight className="h-4 w-4" />
          </Toggle>
          <Toggle
            size="sm"
            pressed={editor.isActive({ textAlign: "justify" })}
            onPressedChange={() =>
              editor.chain().focus().setTextAlign("justify").run()
            }
          >
            <AlignJustify className="h-4 w-4" />
          </Toggle>

          <div className="mx-1 h-6 w-px bg-border" />

          {/* Colors */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <Palette className="h-4 w-4" />
                <span className="sr-only">Text color</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-48 p-2">
              <div className="grid grid-cols-5 gap-1">
                {COLORS.map((color) => (
                  <button
                    key={color.value}
                    type="button"
                    className="h-6 w-6 rounded-full border"
                    style={{ backgroundColor: color.value }}
                    onClick={() => setTextColor(color.value)}
                    title={color.name}
                  />
                ))}
              </div>
            </PopoverContent>
          </Popover>

          {/* Font Sizes */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <Type className="h-4 w-4" />
                <span className="sr-only">Font size</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-32 p-2">
              <div className="space-y-1">
                {FONT_SIZES.map((size) => (
                  <button
                    key={size.value}
                    type="button"
                    className="block w-full rounded px-2 py-1 text-left text-sm hover:bg-muted"
                    onClick={() => setFontSize(size.value)}
                  >
                    {size.name}
                  </button>
                ))}
              </div>
            </PopoverContent>
          </Popover>

          <div className="mx-1 h-6 w-px bg-border" />

          {/* Links */}
          <Popover open={showLinkInput} onOpenChange={setShowLinkInput}>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0"
                onClick={() => {
                  const previousUrl = editor.getAttributes("link").href
                  setLinkUrl(previousUrl || "")
                }}
              >
                <Link2 className="h-4 w-4" />
                <span className="sr-only">Add link</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-4" align="start">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Input
                    placeholder="Paste or type a link..."
                    value={linkUrl}
                    onChange={(e) => setLinkUrl(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault()
                        setLink()
                      }
                    }}
                  />
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => {
                      setLink()
                    }}
                  >
                    Apply
                  </Button>
                </div>
                {editor.isActive("link") && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={() => {
                      editor.chain().focus().unsetLink().run()
                      setShowLinkInput(false)
                    }}
                  >
                    <Unlink className="mr-2 h-4 w-4" />
                    Remove link
                  </Button>
                )}
              </div>
            </PopoverContent>
          </Popover>

          {/* Images */}
          <Popover open={showImageInput} onOpenChange={setShowImageInput}>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <ImageIcon className="h-4 w-4" />
                <span className="sr-only">Add image</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-4" align="start">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Input
                    placeholder="Paste image URL..."
                    value={imageUrl}
                    onChange={(e) => setImageUrl(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault()
                        addImage()
                      }
                    }}
                  />
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => {
                      addImage()
                    }}
                  >
                    Add
                  </Button>
                </div>
              </div>
            </PopoverContent>
          </Popover>

          <div className="mx-1 h-6 w-px bg-border" />

          {/* Undo/Redo */}
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => editor.chain().focus().undo().run()}
            disabled={!editor.can().chain().focus().undo().run()}
          >
            <Undo className="h-4 w-4" />
            <span className="sr-only">Undo</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => editor.chain().focus().redo().run()}
            disabled={!editor.can().chain().focus().redo().run()}
          >
            <Redo className="h-4 w-4" />
            <span className="sr-only">Redo</span>
          </Button>
        </div>
      )}

      <div
        className={cn(
          "prose prose-sm dark:prose-invert max-w-none rounded-b-md border bg-background p-4 focus:outline-none",
          !editable && "rounded-t-md"
        )}
      >
        <EditorContent editor={editor} className="min-h-[200px]" />
      </div>
    </div>
  )
}

// Helper component to render HTML content safely
export function RichTextContent({ content }: { content: string }) {
  return (
    <div
      className="prose prose-sm dark:prose-invert max-w-none"
      dangerouslySetInnerHTML={{ __html: content }}
    />
  )
}
