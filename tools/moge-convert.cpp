#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include "ggml.h"
#include "pt_checkpoint.hpp"

namespace fs = std::filesystem;

namespace {

// ---------------- small JSON parser ----------------
struct Json {
    using array = std::vector<Json>;
    using object = std::map<std::string, Json>;
    using value = std::variant<std::nullptr_t, bool, double, std::string, array, object>;
    value v{};

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_bool() const { return std::holds_alternative<bool>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_array() const { return std::holds_alternative<array>(v); }
    bool is_object() const { return std::holds_alternative<object>(v); }
    const object & obj() const { return std::get<object>(v); }
    const array & arr() const { return std::get<array>(v); }
    const std::string & str() const { return std::get<std::string>(v); }
    bool boolean() const { return std::get<bool>(v); }
    double number() const { return std::get<double>(v); }

    const Json & at(const std::string & k) const {
        auto & o = obj(); auto it = o.find(k);
        if (it == o.end()) throw std::runtime_error("missing JSON key: " + k);
        return it->second;
    }
    const Json * find(const std::string & k) const {
        if (!is_object()) return nullptr;
        auto it = obj().find(k); return it == obj().end() ? nullptr : &it->second;
    }
    int64_t i64() const { return static_cast<int64_t>(std::llround(number())); }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view s) : s_(s) {}
    Json parse() { skip(); Json x = value(); skip(); if (p_ != s_.size()) fail("trailing characters"); return x; }
private:
    std::string_view s_; size_t p_ = 0;
    [[noreturn]] void fail(const std::string & m) const { throw std::runtime_error("JSON parse error at " + std::to_string(p_) + ": " + m); }
    void skip() { while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_; }
    bool take(char c) { skip(); if (p_ < s_.size() && s_[p_] == c) { ++p_; return true; } return false; }
    void expect(char c) { if (!take(c)) fail(std::string("expected '") + c + "'"); }
    Json value() {
        skip(); if (p_ >= s_.size()) fail("unexpected end"); char c=s_[p_];
        if (c=='{') return object(); if (c=='[') return arrayv(); if (c=='\"') return Json{string()};
        if (c=='t') { lit("true"); return Json{true}; }
        if (c=='f') { lit("false"); return Json{false}; }
        if (c=='n') { lit("null"); return Json{nullptr}; }
        if (c=='-' || std::isdigit(static_cast<unsigned char>(c))) return Json{number()};
        fail("invalid value");
    }
    void lit(std::string_view x) { if (s_.substr(p_,x.size()) != x) fail("bad literal"); p_ += x.size(); }
    Json object() {
        expect('{'); Json::object o; skip(); if (take('}')) return Json{o};
        for (;;) { skip(); if (p_>=s_.size() || s_[p_]!='\"') fail("object key must be string"); std::string k=string(); expect(':'); o.emplace(std::move(k), value()); if (take('}')) break; expect(','); }
        return Json{o};
    }
    Json arrayv() {
        expect('['); Json::array a; skip(); if (take(']')) return Json{a};
        for (;;) { a.push_back(value()); if (take(']')) break; expect(','); }
        return Json{a};
    }
    std::string string() {
        expect('\"'); std::string out;
        while (p_ < s_.size()) {
            char c=s_[p_++]; if (c=='\"') return out; if (c!='\\') { out.push_back(c); continue; }
            if (p_>=s_.size()) fail("bad escape"); char e=s_[p_++];
            switch(e) { case '\"':out.push_back('\"');break; case '\\':out.push_back('\\');break; case '/':out.push_back('/');break; case 'b':out.push_back('\b');break; case 'f':out.push_back('\f');break; case 'n':out.push_back('\n');break; case 'r':out.push_back('\r');break; case 't':out.push_back('\t');break;
                case 'u': { if (p_+4>s_.size()) fail("short unicode escape"); unsigned cp=0; for(int i=0;i<4;++i){ char h=s_[p_++]; cp<<=4; if(h>='0'&&h<='9')cp+=h-'0'; else if(h>='a'&&h<='f')cp+=10+h-'a'; else if(h>='A'&&h<='F')cp+=10+h-'A'; else fail("bad unicode escape"); } if(cp<0x80)out.push_back(char(cp)); else if(cp<0x800){out.push_back(char(0xC0|(cp>>6)));out.push_back(char(0x80|(cp&63)));} else {out.push_back(char(0xE0|(cp>>12)));out.push_back(char(0x80|((cp>>6)&63)));out.push_back(char(0x80|(cp&63)));} break; }
                default: fail("unknown escape"); }
        }
        fail("unterminated string");
    }
    double number() {
        skip(); size_t b=p_; if (s_[p_]=='-') ++p_; while(p_<s_.size()&&std::isdigit((unsigned char)s_[p_]))++p_; if(p_<s_.size()&&s_[p_]=='.'){++p_;while(p_<s_.size()&&std::isdigit((unsigned char)s_[p_]))++p_;} if(p_<s_.size()&&(s_[p_]=='e'||s_[p_]=='E')){++p_;if(p_<s_.size()&&(s_[p_]=='+'||s_[p_]=='-'))++p_;while(p_<s_.size()&&std::isdigit((unsigned char)s_[p_]))++p_;}
        std::string tmp(s_.substr(b,p_-b)); char * end=nullptr; double d=std::strtod(tmp.c_str(), &end); if(!end||*end)fail("bad number"); return d;
    }
};

std::string read_text(const fs::path & p) { std::ifstream f(p); if(!f) throw std::runtime_error("cannot open " + p.string()); std::ostringstream ss; ss<<f.rdbuf(); return ss.str(); }
std::vector<uint8_t> read_bytes(const fs::path & p) { std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f)throw std::runtime_error("cannot open "+p.string()); auto n=f.tellg(); std::vector<uint8_t>b((size_t)n); f.seekg(0); if(n>0&&!f.read((char*)b.data(),n))throw std::runtime_error("read failed: "+p.string()); return b; }

// ---------------- safetensors ----------------
struct STensor { std::string name,dtype; std::vector<int64_t> shape; size_t begin=0,end=0; std::optional<moge_pt::Storage> pt_storage; };
struct SafeFile { std::vector<uint8_t> bytes; size_t data_base=0; std::map<std::string,STensor> tensors; std::shared_ptr<moge_pt::Checkpoint> pt; };
uint64_t u64le(const uint8_t *p){uint64_t v=0;for(int i=7;i>=0;--i)v=(v<<8)|p[i];return v;}
SafeFile load_safe(const fs::path & p) {
    SafeFile sf; sf.bytes=read_bytes(p); if(sf.bytes.size()<8)throw std::runtime_error("safetensors file too small"); uint64_t hn=u64le(sf.bytes.data()); if(hn>sf.bytes.size()-8)throw std::runtime_error("bad safetensors header length"); sf.data_base=8+(size_t)hn;
    std::string_view hs((const char*)sf.bytes.data()+8,(size_t)hn); Json root=JsonParser(hs).parse(); if(!root.is_object())throw std::runtime_error("safetensors header is not object");
    for(auto &kv:root.obj()) { if(kv.first=="__metadata__")continue; const Json &j=kv.second; STensor t; t.name=kv.first; t.dtype=j.at("dtype").str(); for(auto &x:j.at("shape").arr()) t.shape.push_back(x.i64()); auto &off=j.at("data_offsets").arr(); if(off.size()!=2)throw std::runtime_error("bad data_offsets for "+t.name); t.begin=(size_t)off[0].i64(); t.end=(size_t)off[1].i64(); if(t.end<t.begin||sf.data_base+t.end>sf.bytes.size())throw std::runtime_error("bad tensor extent: "+t.name); sf.tensors.emplace(t.name,t); }
    return sf;
}

size_t elements(const std::vector<int64_t>&s);
Json json_from_pt(const moge_pt::Value &v) {
    if (std::holds_alternative<std::monostate>(v.v)) return Json{nullptr};
    if (auto x=std::get_if<bool>(&v.v)) return Json{*x};
    if (auto x=std::get_if<int64_t>(&v.v)) return Json{double(*x)};
    if (auto x=std::get_if<double>(&v.v)) return Json{*x};
    if (auto x=std::get_if<std::string>(&v.v)) return Json{*x};
    if (auto x=std::get_if<moge_pt::Value::List>(&v.v)) { Json::array a; for (auto &e:*x) a.push_back(json_from_pt(e)); return Json{a}; }
    if (auto x=std::get_if<moge_pt::Value::Dict>(&v.v)) { Json::object o; for (auto &kv:*x) o.emplace(kv.first,json_from_pt(kv.second)); return Json{o}; }
    throw std::runtime_error("unsupported object inside model_config");
}
SafeFile load_pt(const fs::path &p, Json &cfg) {
    SafeFile sf; sf.pt=std::make_shared<moge_pt::Checkpoint>(p); cfg=json_from_pt(sf.pt->config); sf.data_base=0;
    for (auto &kv: sf.pt->tensors) {
        const auto &td=kv.second; STensor t; t.name=kv.first; t.dtype=td.storage.dtype; t.shape=td.shape; t.pt_storage=td.storage;
        size_t es=(t.dtype=="F32"||t.dtype=="I32")?4:(t.dtype=="F16"||t.dtype=="BF16")?2:(t.dtype=="I64")?8:0;
        if(!es) throw std::runtime_error("unsupported tensor dtype in .pt: "+t.dtype);
        size_t n=elements(t.shape); if(td.offset<0) throw std::runtime_error("negative tensor storage offset");
        uint64_t bo=uint64_t(td.offset)*es, nb=uint64_t(n)*es, avail=sf.pt->zip.size(sf.pt->prefix+"data/"+td.storage.key);
        if(bo+nb>avail) throw std::runtime_error("tensor exceeds storage: "+t.name); t.begin=(size_t)bo; t.end=(size_t)(bo+nb); sf.tensors.emplace(t.name,t);
    }
    return sf;
}
size_t elements(const std::vector<int64_t>&s){size_t n=1;for(auto x:s){if(x<=0||n>std::numeric_limits<size_t>::max()/(size_t)x)throw std::runtime_error("bad tensor shape");n*=size_t(x);}return n;}
float f16_to_f32(uint16_t h){ uint32_t sign=(h&0x8000u)<<16, exp=(h>>10)&31, mant=h&1023; uint32_t u; if(exp==0){ if(mant==0)u=sign; else { int e=-14; while((mant&0x400)==0){mant<<=1;--e;} mant&=0x3ff; u=sign|uint32_t(e+127)<<23|mant<<13; }} else if(exp==31)u=sign|0x7f800000u|mant<<13; else u=sign|((exp-15+127)<<23)|mant<<13; float f;std::memcpy(&f,&u,4);return f; }
uint16_t f32_to_f16(float f){ uint32_t x;std::memcpy(&x,&f,4); uint32_t sign=(x>>16)&0x8000, mant=x&0x7fffff, exp=(x>>23)&0xff; if(exp==255)return uint16_t(sign|0x7c00|(mant?0x200:0)); int e=int(exp)-127+15; if(e>=31)return uint16_t(sign|0x7c00); if(e<=0){ if(e<-10)return uint16_t(sign); mant|=0x800000; int sh=14-e; uint32_t r=mant>>sh, rem=mant&((1u<<sh)-1), half=1u<<(sh-1); if(rem>half||(rem==half&&(r&1)))++r; return uint16_t(sign|r);} uint32_t r=mant>>13, rem=mant&0x1fff; if(rem>0x1000||(rem==0x1000&&(r&1))){++r;if(r==0x400){r=0;++e;if(e>=31)return uint16_t(sign|0x7c00);}} return uint16_t(sign|(uint32_t(e)<<10)|r); }
float bf16_to_f32(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
uint16_t f32_to_bf16(float f){uint32_t u;std::memcpy(&u,&f,4);u+=0x7fffu+((u>>16)&1u);return uint16_t(u>>16);}
std::vector<float> tensor_float(const SafeFile &sf,const STensor&t){size_t n=elements(t.shape);std::vector<uint8_t> raw;const uint8_t*p=nullptr;if(t.pt_storage){raw=sf.pt->storage(*t.pt_storage);p=raw.data()+t.begin;}else p=sf.bytes.data()+sf.data_base+t.begin;size_t bytes=t.end-t.begin;std::vector<float>o(n); if(t.dtype=="F32"){if(bytes!=n*4)throw std::runtime_error("size mismatch "+t.name);for(size_t i=0;i<n;++i)std::memcpy(&o[i],p+4*i,4);} else if(t.dtype=="F16"){if(bytes!=n*2)throw std::runtime_error("size mismatch "+t.name);for(size_t i=0;i<n;++i){uint16_t x;std::memcpy(&x,p+2*i,2);o[i]=f16_to_f32(x);}} else if(t.dtype=="BF16"){if(bytes!=n*2)throw std::runtime_error("size mismatch "+t.name);for(size_t i=0;i<n;++i){uint16_t x;std::memcpy(&x,p+2*i,2);o[i]=bf16_to_f32(x);}} else throw std::runtime_error("floating tensor dtype unsupported: "+t.dtype+" for "+t.name); return o;}
std::vector<int32_t> tensor_i32(const SafeFile&sf,const STensor&t){size_t n=elements(t.shape);std::vector<uint8_t> raw;const uint8_t*p=nullptr;if(t.pt_storage){raw=sf.pt->storage(*t.pt_storage);p=raw.data()+t.begin;}else p=sf.bytes.data()+sf.data_base+t.begin;size_t bytes=t.end-t.begin;std::vector<int32_t>o(n);if(t.dtype=="I32"){if(bytes!=n*4)throw std::runtime_error("size mismatch");std::memcpy(o.data(),p,bytes);}else if(t.dtype=="I64"){if(bytes!=n*8)throw std::runtime_error("size mismatch");for(size_t i=0;i<n;++i){int64_t x;std::memcpy(&x,p+8*i,8);o[i]=(int32_t)x;}}else throw std::runtime_error("integer tensor dtype unsupported: "+t.dtype);return o;}

// ---------------- MOGG writer ----------------
enum : uint16_t { T_F32=1,T_F16=2,T_BF16=3,T_I32=4 };
enum : uint8_t { TF_SENSITIVE=1u<<0,TF_LINEARIZED_1X1=1u<<1,TF_PHASED_CONVT=1u<<2,TF_SPARSE=1u<<3,TF_SPARSE_PACKED=1u<<4,TF_PIXELSHUFFLE_PHASED=1u<<5,TF_HADAMARD_ROTATED=1u<<6 };
using Meta=std::variant<int64_t,double,std::string,std::vector<int64_t>,std::vector<std::string>,bool>;
struct OutTensor {std::string name;std::vector<int64_t> shape;uint16_t type=0;uint8_t flags=0;std::vector<uint8_t>data;};
struct Writer {unsigned alignment=256;std::vector<std::pair<std::string,Meta>>meta;std::vector<OutTensor>tensors;void add(std::string k,Meta v){meta.emplace_back(std::move(k),std::move(v));}};
uint64_t align_up(uint64_t n,uint64_t a){return(n+a-1)/a*a;}
void append(std::vector<uint8_t>&b,const void*p,size_t n){auto*q=(const uint8_t*)p;b.insert(b.end(),q,q+n);}template<class T>void pod(std::vector<uint8_t>&b,T x){append(b,&x,sizeof x);}void p16(std::vector<uint8_t>&b,const std::string&s){if(s.size()>65535)throw std::runtime_error("string too long");pod<uint16_t>(b,(uint16_t)s.size());append(b,s.data(),s.size());}
std::pair<uint8_t,std::vector<uint8_t>> meta_payload(const Meta&m){std::vector<uint8_t>b;if(auto*p=std::get_if<bool>(&m)){pod<uint8_t>(b,*p?1:0);return{6,b};}if(auto*p=std::get_if<int64_t>(&m)){pod<int64_t>(b,*p);return{1,b};}if(auto*p=std::get_if<double>(&m)){pod<double>(b,*p);return{2,b};}if(auto*p=std::get_if<std::string>(&m)){pod<uint32_t>(b,(uint32_t)p->size());append(b,p->data(),p->size());return{3,b};}if(auto*p=std::get_if<std::vector<int64_t>>(&m)){pod<uint32_t>(b,(uint32_t)p->size());for(auto x:*p)pod<int64_t>(b,x);return{4,b};}auto*p=std::get_if<std::vector<std::string>>(&m);pod<uint32_t>(b,(uint32_t)p->size());for(auto&s:*p)p16(b,s);return{5,b};}
std::vector<uint8_t> index_bytes(const Writer&w,const std::vector<uint64_t>&offs){std::vector<uint8_t>b;for(size_t i=0;i<w.tensors.size();++i){auto&t=w.tensors[i];p16(b,t.name);pod<uint16_t>(b,t.type);pod<uint8_t>(b,(uint8_t)t.shape.size());pod<uint8_t>(b,t.flags);std::array<int64_t,4>ne{1,1,1,1};for(size_t d=0;d<t.shape.size();++d)ne[d]=t.shape[t.shape.size()-1-d];for(auto x:ne)pod<int64_t>(b,x);pod<uint64_t>(b,offs[i]);pod<uint64_t>(b,t.data.size());}return b;}
void write_mogg(const Writer&w,const fs::path&p){std::vector<uint8_t>mb;for(auto&kv:w.meta){auto [tag,pl]=meta_payload(kv.second);p16(mb,kv.first);pod<uint8_t>(mb,tag);pod<uint32_t>(mb,(uint32_t)pl.size());append(mb,pl.data(),pl.size());}uint64_t header=40,meta_off=header;std::vector<uint64_t>dummy(w.tensors.size());auto idx0=index_bytes(w,dummy);uint64_t data_start=align_up(meta_off+mb.size()+idx0.size(),w.alignment),cur=data_start;std::vector<uint64_t>offs;for(auto&t:w.tensors){cur=align_up(cur,w.alignment);offs.push_back(cur);cur+=t.data.size();}auto idx=index_bytes(w,offs);std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot write "+p.string());const uint8_t magic[8]={'M','O','G','G',0,0,0,1};f.write((char*)magic,8);uint32_t ver=1,nm=(uint32_t)w.meta.size(),nt=(uint32_t)w.tensors.size(),al=w.alignment;f.write((char*)&ver,4);f.write((char*)&nm,4);f.write((char*)&nt,4);f.write((char*)&al,4);f.write((char*)&meta_off,8);f.write((char*)&data_start,8);f.write((char*)mb.data(),mb.size());f.write((char*)idx.data(),idx.size());uint64_t pos=(uint64_t)f.tellp();std::vector<char>zero(4096);while(pos<data_start){size_t n=(size_t)std::min<uint64_t>(zero.size(),data_start-pos);f.write(zero.data(),n);pos+=n;}for(size_t i=0;i<w.tensors.size();++i){pos=(uint64_t)f.tellp();while(pos<offs[i]){size_t n=(size_t)std::min<uint64_t>(zero.size(),offs[i]-pos);f.write(zero.data(),n);pos+=n;}auto&d=w.tensors[i].data;f.write((char*)d.data(),d.size());}}

// ---------------- conversion helpers ----------------
struct Args { fs::path input,output; std::string quant="q8"; unsigned alignment=256; };
std::vector<int64_t> ints(const Json&j){std::vector<int64_t>o;for(auto&x:j.arr())o.push_back(x.i64());return o;}
std::vector<std::string> strings(const Json&j){std::vector<std::string>o;for(auto&x:j.arr())o.push_back(x.str());return o;}
std::vector<int64_t> list_or(const Json* j,size_t n,int64_t def=-1){if(!j||j->is_null())return std::vector<int64_t>(n,def);if(j->is_array()){if(j->arr().size()!=n)throw std::runtime_error("config list length mismatch");std::vector<int64_t>o;for(auto&x:j->arr())o.push_back(x.is_null()?def:x.i64());return o;}return std::vector<int64_t>(n,j->i64());}
std::vector<std::string> strlist(const Json&j,size_t n){if(j.is_string())return std::vector<std::string>(n,j.str());auto o=strings(j);if(o.size()!=n)throw std::runtime_error("config string-list length mismatch");return o;}
std::string sget(const Json&o,const std::string&k,const std::string&d){auto*j=o.find(k);return j?j->str():d;}int64_t iget(const Json&o,const std::string&k,int64_t d){auto*j=o.find(k);return j?j->i64():d;}
void add_stack(Writer&w,const std::string&name,const Json*cfg){w.add(name+".present",cfg!=nullptr);if(!cfg)return;auto dims=ints(cfg->at("dim_res_blocks"));size_t n=dims.size();w.add(name+".dim_in",list_or(cfg->find("dim_in"),n));w.add(name+".dim_res_blocks",dims);w.add(name+".dim_out",list_or(cfg->find("dim_out"),n));w.add(name+".resamplers",strlist(cfg->at("resamplers"),n-1));w.add(name+".dim_times_res_block_hidden",iget(*cfg,"dim_times_res_block_hidden",1));w.add(name+".num_res_blocks",list_or(cfg->find("num_res_blocks"),n,1));w.add(name+".res_block_in_norm",sget(*cfg,"res_block_in_norm","layer_norm"));w.add(name+".res_block_hidden_norm",sget(*cfg,"res_block_hidden_norm","group_norm"));w.add(name+".activation",sget(*cfg,"activation","relu"));}
bool starts(const std::string&s,const std::string&p){return s.rfind(p,0)==0;}bool ends(const std::string&s,const std::string&p){return s.size()>=p.size()&&s.compare(s.size()-p.size(),p.size(),p)==0;}
std::optional<std::string> resampler_kind(const std::string&name,const Json&cfg){for(auto stack:{"neck","points_head","normal_head","mask_head"}){std::string p=std::string(stack)+".resamplers.";if(!starts(name,p))continue;auto rest=name.substr(p.size());auto dot=rest.find('.');if(dot==std::string::npos)return{};int idx=std::stoi(rest.substr(0,dot));auto*sc=cfg.find(stack);if(!sc)return{};auto dims=ints(sc->at("dim_res_blocks"));auto rs=strlist(sc->at("resamplers"),dims.size()-1);if(idx<0||size_t(idx)>=rs.size())return{};return rs[idx];}return{};}
bool sensitive(const std::string&name,const Json&cfg){if(name=="encoder.backbone.pos_embed")return true;for(auto stack:{"points_head","normal_head"}){auto*hc=cfg.find(stack);if(!hc||!starts(name,std::string(stack)+".output_blocks."))continue;auto dims=ints(hc->at("dim_res_blocks"));std::string p=std::string(stack)+".output_blocks."+std::to_string(dims.size()-1)+".";if(starts(name,p))return true;}return false;}
void permute_convt(std::vector<float>&x,std::vector<int64_t>&sh){if(sh.size()!=4||sh[2]!=2||sh[3]!=2)throw std::runtime_error("bad convtranspose shape");int64_t ci=sh[0],co=sh[1];std::vector<float>y(x.size());for(int64_t i=0;i<ci;++i)for(int64_t o=0;o<co;++o)for(int64_t ky=0;ky<2;++ky)for(int64_t kx=0;kx<2;++kx)y[(((ky*2+kx)*co+o)*ci)+i]=x[(((i*co+o)*2+ky)*2+kx)];x.swap(y);sh={co*4,ci};}
void phase_pixel(std::vector<float>&x,std::vector<int64_t>&sh){if(sh.empty()||sh[0]%4)throw std::runtime_error("bad pixelshuffle channels");int64_t c=sh[0]/4,rest=1;for(size_t i=1;i<sh.size();++i)rest*=sh[i];std::vector<float>y(x.size());for(int64_t oc=0;oc<c;++oc)for(int p=0;p<4;++p)for(int64_t r=0;r<rest;++r)y[((p*c+oc)*rest)+r]=x[((oc*4+p)*rest)+r];x.swap(y);}
std::pair<uint16_t,std::vector<uint8_t>> encode_float(const std::vector<float>&x,const std::string&dt){std::vector<uint8_t>b;if(dt=="f32"){b.resize(x.size()*4);std::memcpy(b.data(),x.data(),b.size());return{T_F32,b};}b.resize(x.size()*2);for(size_t i=0;i<x.size();++i){uint16_t q=f32_to_f16(x[i]);std::memcpy(b.data()+2*i,&q,2);}return{T_F16,b};}
std::vector<uint8_t> encode_q8(const std::vector<float>&x,int64_t rows,int64_t cols){
    // Preserve the validated production recipe exactly: checkpoint F32 -> F16
    // storage rounding -> FP32 expansion -> ggml Q8_0.  Quantizing the original
    // checkpoint F32 values directly produces a different model than the release
    // production artifact even though it uses the same Q8_0 kernel.
    std::vector<ggml_fp16_t> h(x.size());
    std::vector<float> rounded(x.size());
    ggml_fp32_to_fp16_row(x.data(), h.data(), x.size());
    ggml_fp16_to_fp32_row(h.data(), rounded.data(), x.size());
    size_t sz=ggml_row_size(GGML_TYPE_Q8_0,cols)*size_t(rows);
    std::vector<uint8_t>b(sz);
    size_t wrote=ggml_quantize_chunk(GGML_TYPE_Q8_0,rounded.data(),b.data(),0,rows,cols,nullptr);
    if(wrote!=sz||!ggml_validate_row_data(GGML_TYPE_Q8_0,b.data(),b.size()))throw std::runtime_error("Q8_0 quantization validation failed");
    return b;
}
const char * usage_text() {
#ifdef MOGE_PROJECT_VERSION_STRING
    return "moge-convert " MOGE_PROJECT_VERSION_STRING "\nConvert an upstream MoGe PyTorch checkpoint to .moge.\n\nUSAGE\n  moge-convert MODEL.pt [--quant q8|f16|f32] [--output FILE.moge]\n\nEXAMPLE\n  moge-convert model.pt\n  # writes v3_q8.moge\n\nOPTIONS\n  -q, --quant <type>    Output precision: q8, f16, or f32. Default: q8.\n  -o, --output <file>   Output path. Default: v{model_version}_{quant}.moge.\n  --version             Show version and exit.\n  -h, --help            Show this help and exit.";
#else
    return "moge-convert\nUSAGE\n  moge-convert MODEL.pt [--quant q8|f16|f32] [--output FILE.moge]";
#endif
}
Args parse_args(int argc,char**argv){if(argc<2)throw std::runtime_error("missing MODEL.pt (run --help for usage)");Args a;for(int i=1;i<argc;++i){std::string s=argv[i];auto need=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+s);return std::string(argv[i]);};if(s=="-q"||s=="--quant")a.quant=need();else if(s=="-o"||s=="--output")a.output=need();else if(!s.empty()&&s[0]=='-')throw std::runtime_error("unknown option: "+s);else if(a.input.empty())a.input=s;else throw std::runtime_error("unexpected argument: "+s);}if(a.input.empty())throw std::runtime_error("missing MODEL.pt");if(a.input.extension()!=".pt"&&a.input.extension()!=".pth")throw std::runtime_error("input must be an upstream .pt or .pth checkpoint");if(a.quant!="q8"&&a.quant!="f16"&&a.quant!="f32")throw std::runtime_error("--quant must be q8, f16, or f32");return a;}

void convert(Args a){Json cfg;SafeFile sf=load_pt(a.input,cfg);if(auto*m=cfg.find("model");m&&m->is_object())cfg=*m;if(!cfg.is_object())throw std::runtime_error("embedded model_config must be an object");int ver=2;for(auto&kv:sf.tensors)if(starts(kv.first,"refiner.")){ver=3;break;}if(a.output.empty())a.output="v"+std::to_string(ver)+"_"+a.quant+".moge";
    auto clsit=sf.tensors.find("encoder.backbone.cls_token");if(clsit==sf.tensors.end()||clsit->second.shape.empty())throw std::runtime_error("missing encoder.backbone.cls_token");int64_t embed=clsit->second.shape.back(),depth=0;for(auto&kv:sf.tensors){std::string p="encoder.backbone.blocks.";if(starts(kv.first,p)){auto tail=kv.first.substr(p.size());auto dot=tail.find('.');if(dot!=std::string::npos)depth=std::max<int64_t>(depth,std::stoll(tail.substr(0,dot))+1);}}if(!depth)throw std::runtime_error("no DINO blocks found");int64_t heads=embed/64;std::string ffn=sf.tensors.count("encoder.backbone.blocks.0.mlp.w12.weight")?"swiglu":"gelu";
    const Json&ecfg=cfg.at("encoder");std::vector<int64_t>inter;if(auto*j=ecfg.find("intermediate_layers")){if(j->is_array())inter=ints(*j);else{int64_t n=j->i64();for(int64_t i=depth-n;i<depth;++i)inter.push_back(i);}}else throw std::runtime_error("missing encoder.intermediate_layers");
    Writer w;w.alignment=a.alignment;w.add("format.name",std::string("MOGG"));w.add("format.version",int64_t(1));w.add("format.storage",a.quant=="q8"?std::string("q8_0"):a.quant);w.add("format.storage.profile",std::string("default"));w.add("format.storage.encoder",a.quant);w.add("format.storage.decoder",a.quant);w.add("format.storage.refiner",a.quant=="q8"?std::string("f16"):a.quant);w.add("format.quantization",a.quant=="q8"?std::string("q8_0"):std::string("none"));w.add("format.quantization.importance",false);w.add("format.prepack.1x1",true);w.add("format.prepack.convtranspose_phase",true);w.add("format.prepack.sparse27",ver==3);w.add("format.prepack.pixelshuffle_phase",true);w.add("format.rotation",std::string("none"));w.add("format.rotation.group",int64_t(0));w.add("source.name",a.input.filename().string());w.add("model.version",int64_t(ver));w.add("model.remap_output",ver==3?std::string("exp"):sget(cfg,"remap_output","linear"));if(auto*j=cfg.find("num_tokens_range"))w.add("model.num_tokens_range",ints(*j));else w.add("model.num_tokens_range",std::vector<int64_t>{1200,3600});w.add("encoder.backbone",ecfg.at("backbone").str());w.add("encoder.embed_dim",embed);w.add("encoder.depth",depth);w.add("encoder.num_heads",heads);w.add("encoder.dim_out",ecfg.at("dim_out").i64());w.add("encoder.ffn",ffn);w.add("encoder.intermediate_layers",inter);for(auto n:{"neck","points_head","normal_head","mask_head"})add_stack(w,n,cfg.find(n));auto*sh=cfg.find("scale_head");w.add("scale_head.present",sh!=nullptr);if(sh)w.add("scale_head.dims",ints(sh->at("dims")));const Json*rc=(ver==3)?cfg.find("refiner"):nullptr;w.add("refiner.present",rc!=nullptr);if(rc){auto ch=ints(rc->at("model_channels"));w.add("refiner.in_channels",iget(*rc,"in_channels",3));w.add("refiner.out_channels",iget(*rc,"out_channels",1));w.add("refiner.encoder_channels",rc->at("encoder_channels").i64());w.add("refiner.model_channels",ch);w.add("refiner.encoder_blocks_per_level",list_or(rc->find("encoder_blocks_per_level"),ch.size(),1));w.add("refiner.decoder_blocks_per_level",list_or(rc->find("decoder_blocks_per_level"),ch.size()-1,1));w.add("refiner.bottleneck_blocks",iget(*rc,"bottleneck_blocks",1));w.add("refiner.downsample_factors",list_or(rc->find("downsample_factors"),ch.size()-1,2));w.add("refiner.encoder_downsample",iget(*rc,"encoder_downsample",16));w.add("refiner.depth_resolution",cfg.find("refiner_depth_resolution")?cfg.at("refiner_depth_resolution").number():256.0);}
    std::map<std::string,int64_t>stats{{"linearized_1x1",0},{"phase_convt",0},{"sparse27",0},{"pixelshuffle",0},{"hadamard_rotated",0},{"sensitive_f32",0}};std::map<std::string,bool>seen;
    for(auto&kv:sf.tensors){const STensor&t=kv.second;bool isint=t.dtype=="I32"||t.dtype=="I64";bool isfloat=t.dtype=="F32"||t.dtype=="F16"||t.dtype=="BF16";if(!isint&&!isfloat)continue;OutTensor o;o.name=t.name;o.shape=t.shape;auto kind=resampler_kind(t.name,cfg);if(isint){auto x=tensor_i32(sf,t);if(ver==3&&starts(t.name,"refiner."))o.flags|=TF_SPARSE;o.type=T_I32;o.data.resize(x.size()*4);std::memcpy(o.data.data(),x.data(),o.data.size());w.tensors.push_back(std::move(o));seen[t.name]=true;continue;}auto x=tensor_float(sf,t);if(kind&&*kind=="conv_transpose"&&ends(t.name,".0.weight")){permute_convt(x,o.shape);o.name+=".mogg_phase";o.flags|=TF_PHASED_CONVT;++stats["phase_convt"];}
        else {if(kind&&*kind=="pixel_shuffle"&&(ends(t.name,".0.weight")||ends(t.name,".0.bias"))){phase_pixel(x,o.shape);o.flags|=TF_PIXELSHUFFLE_PHASED;++stats["pixelshuffle"];}if(o.shape.size()==4&&o.shape[2]==1&&o.shape[3]==1){o.shape={o.shape[0],o.shape[1]};o.flags|=TF_LINEARIZED_1X1;++stats["linearized_1x1"];}if(ver==3&&starts(t.name,"refiner.")&&t.shape.size()==5&&t.shape[1]==3&&t.shape[2]==3&&t.shape[3]==3){o.shape={t.shape[0],27*t.shape[4]};o.name+=".mogg_sparse27";o.flags|=TF_SPARSE|TF_SPARSE_PACKED;++stats["sparse27"];}}
        if(ver==3&&starts(t.name,"refiner."))o.flags|=TF_SPARSE;bool sens=sensitive(t.name,cfg);if(sens)o.flags|=TF_SENSITIVE;bool keep=sens;if(keep)++stats["sensitive_f32"];bool q8ok=a.quant=="q8"&&!sens&&!(o.flags&TF_SPARSE)&&o.shape.size()==2&&elements(o.shape)>=4096&&o.shape.back()%32==0;if(q8ok){o.type=12;o.data=encode_q8(x,o.shape[0],o.shape[1]);}else{auto enc=encode_float(x,keep?"f32":(a.quant=="f32"?"f32":"f16"));o.type=enc.first;o.data=std::move(enc.second);}if(seen[o.name])throw std::runtime_error("duplicate converted name: "+o.name);seen[o.name]=true;w.tensors.push_back(std::move(o));}
    for(auto pair:{std::pair<const char*,std::array<float,3>>{"encoder.image_mean",{0.485f,0.456f,0.406f}}, {"encoder.image_std",{0.229f,0.224f,0.225f}}})if(!seen[pair.first]){OutTensor o;o.name=pair.first;o.shape={1,3,1,1};std::vector<float>x(pair.second.begin(),pair.second.end());auto enc=encode_float(x,"f32");o.type=enc.first;o.data=std::move(enc.second);w.tensors.push_back(std::move(o));}
    write_mogg(w,a.output);uint64_t payload=0;for(auto&t:w.tensors)payload+=t.data.size();std::cout<<"{\n  \"output\": \""<<a.output.string()<<"\",\n  \"model_version\": "<<ver<<",\n  \"backbone\": \""<<ecfg.at("backbone").str()<<"\",\n  \"embed_dim\": "<<embed<<",\n  \"depth\": "<<depth<<",\n  \"heads\": "<<heads<<",\n  \"ffn\": \""<<ffn<<"\",\n  \"tensors\": "<<w.tensors.size()<<",\n  \"payload_mib\": "<<std::fixed<<std::setprecision(2)<<(double(payload)/(1u<<20));for(auto&s:stats)std::cout<<",\n  \""<<s.first<<"\": "<<s.second;std::cout<<"\n}\n";
}

} // namespace

int main(int argc,char**argv){
    if(argc==2 && (std::string(argv[1])=="--help" || std::string(argv[1])=="-h")){std::cout<<usage_text()<<"\n";return 0;}
#ifdef MOGE_PROJECT_VERSION_STRING
    if(argc==2 && std::string(argv[1])=="--version"){std::cout<<"moge-convert "<<MOGE_PROJECT_VERSION_STRING<<"\n";return 0;}
#endif
    try{Args a=parse_args(argc,argv);convert(a);return 0;}catch(const std::exception&e){std::cerr<<"moge-convert: "<<e.what()<<"\n";return 2;}
}
