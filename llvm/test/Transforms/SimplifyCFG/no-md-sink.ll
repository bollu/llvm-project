; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -sink-common-insts -S | FileCheck %s
; RUN: opt < %s -passes='simplifycfg<sink-common-insts>' -S | FileCheck %s

define i1 @test1(i1 zeroext %flag, i8* %y) #0 {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[S:%.*]] = call i1 @llvm.type.test(i8* [[Y:%.*]], metadata [[META0:![0-9]+]])
; CHECK-NEXT:    [[R:%.*]] = call i1 @llvm.type.test(i8* [[Y]], metadata [[META1:![0-9]+]])
; CHECK-NEXT:    [[T:%.*]] = select i1 [[FLAG:%.*]], i1 [[R]], i1 [[S]]
; CHECK-NEXT:    ret i1 [[T]]
;
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %r = call i1 @llvm.type.test(i8* %y, metadata !0)
  br label %if.end

if.else:
  %s = call i1 @llvm.type.test(i8* %y, metadata !1)
  br label %if.end

if.end:
  %t = phi i1 [ %s, %if.else ], [ %r, %if.then ]
  ret i1 %t
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @test2(i1 zeroext %flag, i8* %y, i8* %z) #0 {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[S:%.*]] = call i1 @llvm.type.test(i8* [[Z:%.*]], metadata [[META1]])
; CHECK-NEXT:    [[R:%.*]] = call i1 @llvm.type.test(i8* [[Y:%.*]], metadata [[META1]])
; CHECK-NEXT:    [[T:%.*]] = select i1 [[FLAG:%.*]], i1 [[R]], i1 [[S]]
; CHECK-NEXT:    ret i1 [[T]]
;
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %r = call i1 @llvm.type.test(i8* %y, metadata !0)
  br label %if.end

if.else:
  %s = call i1 @llvm.type.test(i8* %z, metadata !0)
  br label %if.end

if.end:
  %t = phi i1 [ %s, %if.else ], [ %r, %if.then ]
  ret i1 %t
}
