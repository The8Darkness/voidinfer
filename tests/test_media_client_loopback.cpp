// Deferred public-media loopback qualification. This source intentionally owns
// no server/model lifecycle; an independently started, explicitly qualified
// server is a prerequisite. It must never be a default CTest invocation.
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using Json=nlohmann::json;

namespace {
void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}

Json media_request(std::string_view model,std::string_view media_type,
    std::string_view field,std::string_view locator,bool stream,int maximum) {
    // Text and part ordering are immutable test inputs. Repeated/changed media
    // requests differ only in locator bytes, never in template text or options.
    const std::string prefix=
        "Exact media-cache fixture. Preserve part order and answer with one ASCII word. "
        "This deliberately repeated prefix is long enough to expose a typed cache boundary. "
        "Do not infer audio, metadata, hidden text, or facts outside the supplied media. ";
    return Json{{"model",model},{"store",false},{"stream",stream},
        {"temperature",0.0},{"max_output_tokens",maximum},
        {"input",Json::array({Json{{"type","message"},{"role","user"},
            {"content",Json::array({
                Json{{"type","input_text"},{"text",prefix}},
                Json{{"type",media_type},{std::string(field),locator}},
                Json{{"type","input_text"},{"text"," Return the word now."}}
            })}}})}};
}

Json post_json(httplib::Client& client,std::string_view path,const Json& body) {
    const auto response=client.Post(std::string(path),body.dump(),"application/json");
    need(bool(response),"loopback request transport failed");
    need(response->status==200,"loopback request returned HTTP "+
        std::to_string(response->status)+": "+response->body);
    return Json::parse(response->body);
}

int count_tokens(httplib::Client& client,const Json& request) {
    Json body=request;
    body.erase("store");body.erase("stream");body.erase("temperature");
    body.erase("max_output_tokens");
    const auto result=post_json(client,"/v1/responses/input_tokens",body);
    need(result.at("object")=="response.input_tokens","input-token object kind");
    return result.at("input_tokens").get<int>();
}

struct Terminal {
    Json body;
    int input_tokens=0,cached_tokens=0,output_tokens=0;
};

Terminal terminal(httplib::Client& client,const Json& request) {
    const auto body=post_json(client,"/v1/responses",request);
    need(body.at("object")=="response","terminal response object kind");
    const auto& usage=body.at("usage");
    Terminal result{body,usage.at("input_tokens").get<int>(),
        usage.at("input_tokens_details").at("cached_tokens").get<int>(),
        usage.at("output_tokens").get<int>()};
    need(result.cached_tokens>=0 && result.cached_tokens<=result.input_tokens,
        "typed cache usage exceeds prompt extent");
    return result;
}

std::vector<Json> parse_sse(std::string_view wire) {
    std::vector<Json> events;
    std::size_t cursor=0;
    while(cursor<wire.size()) {
        const auto end=wire.find("\n\n",cursor);
        need(end!=std::string_view::npos,"unterminated SSE event");
        const auto frame=wire.substr(cursor,end-cursor);
        const auto line=frame.find('\n');
        need(line!=std::string_view::npos && frame.substr(0,7)=="event: ",
            "invalid SSE event field");
        const auto type=frame.substr(7,line-7);
        need(frame.substr(line+1,6)=="data: ","invalid SSE data field");
        auto value=Json::parse(frame.substr(line+7));
        need(value.at("type").get<std::string>()==type,"SSE event/data type mismatch");
        events.push_back(std::move(value));cursor=end+2;
    }
    return events;
}

std::vector<Json> stream_all(httplib::Client& client,const Json& body) {
    httplib::Request request;
    request.method="POST";request.path="/v1/responses";
    request.headers.emplace("Content-Type","application/json");
    request.body=body.dump();
    std::string wire;bool headers=false;
    request.response_handler=[&](const httplib::Response& response) {
        headers=response.status==200 && response.get_header_value("Content-Type").find(
            "text/event-stream")!=std::string::npos &&
            response.get_header_value("Cache-Control")=="no-cache";
        return true;
    };
    request.content_receiver=[&](const char* data,std::size_t size,std::uint64_t,std::uint64_t) {
        wire.append(data,size);return true;
    };
    const auto response=client.send(request);
    need(bool(response) && headers,"stream response headers/transport");
    auto events=parse_sse(wire);
    need(!events.empty() && events.front().at("type")=="response.created",
        "stream did not start with response.created");
    std::uint64_t sequence=0;
    for(const auto& event:events)
        need(event.at("sequence_number").get<std::uint64_t>()==sequence++,
            "noncontiguous media SSE sequence");
    return events;
}

void disconnect_after_generation_starts(httplib::Client& client,const Json& body) {
    httplib::Request request;
    request.method="POST";request.path="/v1/responses";
    request.headers.emplace("Content-Type","application/json");request.body=body.dump();
    std::string wire;bool cancelled=false;
    request.content_receiver=[&](const char* data,std::size_t size,std::uint64_t,std::uint64_t) {
        wire.append(data,size);
        if(wire.find("response.output_item.added")!=std::string::npos ||
           wire.find("response.output_text.delta")!=std::string::npos) {
            cancelled=true;return false;
        }
        return true;
    };
    (void)client.send(request);
    need(cancelled,"disconnect fixture ended before generation became observable");
}

void require_output_limit(const Terminal& value) {
    need(value.body.at("status")=="incomplete" && value.output_tokens==1 &&
        value.body.at("incomplete_details").at("reason")=="max_output_tokens",
        "media response changed exact output-limit stop behavior");
}
}

int main(int argc,char** argv) {
    if(argc<8) {
        std::cerr<<"usage: test_media_client_loopback <host> <port> <model> "
            "<image-url-a> <image-url-b-same-geometry> <video-url> "
            "<full|after-reload> [expected-input-tokens]\n";
        return 64;
    }
    httplib::Client client(argv[1],std::stoi(argv[2]));
    client.set_connection_timeout(10);client.set_read_timeout(600);
    const std::string_view model=argv[3],phase=argv[7];
    const auto image_a=media_request(model,"input_image","image_url",argv[4],false,1);
    const auto image_b=media_request(model,"input_image","image_url",argv[5],false,1);
    const auto video=media_request(model,"input_video","video_url",argv[6],false,1);

    const int image_tokens=count_tokens(client,image_a);
    need(image_tokens==count_tokens(client,image_a),
        "identical media changed tokenizer/template token count");
    need(image_tokens==count_tokens(client,image_b),
        "same-geometry changed image changed tokenizer/template extent");

    if(phase=="after-reload") {
        need(argc==9 && image_tokens==std::stoi(argv[8]),
            "reload changed exact tokenizer/template extent");
        const auto first=terminal(client,image_a);require_output_limit(first);
        need(first.cached_tokens==0,"freshly reloaded process retained a typed media root");
        std::cout<<"media_client_after_reload PREPARED_NOT_RUN\n";return 0;
    }
    need(phase=="full","unknown loopback phase");

    const auto first=terminal(client,image_a);require_output_limit(first);
    const auto repeated=terminal(client,image_a);require_output_limit(repeated);
    need(repeated.input_tokens==first.input_tokens &&
        repeated.cached_tokens>first.cached_tokens,
        "repeated identical image did not expose stronger typed reuse");
    const auto changed=terminal(client,image_b);require_output_limit(changed);
    need(changed.input_tokens==repeated.input_tokens &&
        changed.cached_tokens<repeated.cached_tokens,
        "changed image with identical text reused the complete typed root");

    const auto video_result=terminal(client,video);require_output_limit(video_result);
    need(video_result.input_tokens==count_tokens(client,video),
        "video request/template count disagreed with execution");

    Json streaming=image_a;streaming["stream"]=true;
    const auto events=stream_all(client,streaming);
    need(events.back().at("type")=="response.incomplete" &&
        events.back().at("response").at("incomplete_details").at("reason")==
            "max_output_tokens",
        "stream did not preserve output-limit terminal semantics");
    streaming["max_output_tokens"]=64;
    disconnect_after_generation_starts(client,streaming);
    const auto isolated=terminal(client,image_a);require_output_limit(isolated);
    need(isolated.input_tokens==image_tokens,
        "disconnect poisoned subsequent media tokenizer/typed state");

    std::cout<<"reload receipt expected-input-tokens="<<image_tokens<<'\n';
    std::cout<<"media_client_full PREPARED_NOT_RUN\n";
}
