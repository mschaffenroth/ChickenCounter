FROM nginx:1.19-alpine
RUN apk add ffmpeg
ENV SOURCE_ADDRESS=tcp://localhost:9999
ENTRYPOINT nginx && ffmpeg -y -i $SOURCE_ADDRESS  -map 0 -codec:v libx264 -codec:a libfaac -f ssegment -segment_list /usr/share/nginx/html/stream.m3u8 -segment_list_flags +live -segment_time 10 out%03d.ts