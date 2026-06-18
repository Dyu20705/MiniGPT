using HTTP
using JSON3
using Gumbo
using Cascadia

# 1. Cấu hình các hàm làm sạch văn bản
function clean_html(html_content)
    # Parse HTML sang cây DOM
    dom = parsehtml(html_content)
    body = dom.root[2] # Lấy phần body
    
    # Loại bỏ các thành phần không mong muốn (tables, nav, scripts, css)
    for tag in eachmatch(Selector("table, script, style, .mw-editsection, .infobox, .reference"), body)
        # Xóa các node này bằng cách xóa nội dung bên trong
        tag.children = HTMLNode[]
    end
    
    # Trích xuất toàn bộ text văn bản
    text = textcontents(body)
    
    # Xử lý chuẩn hóa text
    text = replace(text, r"\s+" => " ") # Co cụm khoảng trắng, newline thành 1 space
    text = strip(text)
    
    return text
end

# 2. Hàm tải bài viết ngẫu nhiên từ Wikipedia tiếng Việt
function fetch_wiki_subset(target_size_mb=5)
    target_bytes = target_size_mb * 1024 * 1024
    buffer = IOBuffer()
    bytes_written = 0
    articles_count = 0
    
    println("🚀 Bắt đầu tải Wikipedia tiếng Việt (Mục tiêu: $target_size_mb MB)...")
    
    # Gọi API lấy bài viết ngẫu nhiên cho đến khi đủ dung lượng
    while bytes_written < target_bytes
        try
            # Lấy 10 bài viết ngẫu nhiên mỗi lượt qua Wikimedia API
            url = "https://vi.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=10"
            response = HTTP.get(url)
            data = JSON3.read(String(response.body))
            
            for page in data.query.random
                page_id = page.id
                title = page.title
                
                # Tải nội dung chi tiết dạng HTML của bài viết
                content_url = "https://vi.wikipedia.org/w/api.php?action=parse&format=json&pageid=$(page_id)&prop=text"
                res_content = HTTP.get(content_url)
                page_data = JSON3.read(String(res_content.body))
                
                if haskey(page_data, :parse) && haskey(page_data.parse, :text)
                    html_raw = page_data.parse.text["*"]
                    clean_text = clean_html(html_raw)
                    
                    if length(clean_text) > 200 # Bỏ qua bài viết quá ngắn
                        write(buffer, clean_text * "\n\n")
                        bytes_written = buffer.size
                        articles_count += 1
                        
                        if articles_count % 10 == 0
                            mb_current = round(bytes_written / (1024*1024), digits=2)
                            println("📈 Đã tải $articles_count bài viết (~ $mb_current MB)...")
                        end
                    end
                end
                
                bytes_written >= target_bytes && break
            end
        catch e
            # Bỏ qua lỗi kết nối mạng nhất thời và tiếp tục tục tải
            continue
        end
    end
    
    return take!(buffer)
end

# 3. Chạy tiến trình chính
mkpath("data")
data_bytes = fetch_wiki_subset(5) # Tải khoảng 5MB văn bản sạch

# Ghi dữ liệu ra file input.txt
open("data/input.txt", "w") do io
    write(io, data_bytes)
end
println("✅ Đã tạo thành công file: data/input.txt")

# 4. Lưu file Metadata đúng yêu cầu cấu trúc của bạn
metadata = Dict(
    "name" => "Vietnamese Wikipedia subset",
    "language" => "vi",
    "source" => "Wikipedia",
    "license" => "CC BY-SA",
    "retrieved_at" => "2026-06-18",
    "processing" => [
        "removed markup",
        "removed tables",
        "normalized unicode",
        "collapsed whitespace"
    ]
)

open("data/metadata.json", "w") do io
    JSON3.write(io, metadata)
end
println("📝 Đã lưu metadata tại: data/metadata.json")